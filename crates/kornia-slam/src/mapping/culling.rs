//! Landmark culling policy.
//!
//! Selection lives here because the thresholds are a mapping judgement about
//! which landmarks have earned their place. Removal itself is the map's
//! canonical operation, so this selects ids and delegates — there is no second
//! cleanup loop, and the map never calls back into mapping.

use crate::mapping::map::Map;
use std::collections::HashSet;

/// Minimum times a landmark must have been in view before its match rate is
/// judged at all.
const MIN_OBSERVATIONS: u32 = 5;
/// Match rate below which a sufficiently-observed landmark is dropped.
const MIN_FOUND_RATIO: f64 = 0.20;

/// Ids of the landmarks that fail the culling policy.
///
/// Two criteria: a poor found ratio once a landmark has been seen
/// enough times to judge, and a landmark sitting behind its reference keyframe.
fn select_for_culling(map: &Map) -> Vec<usize> {
    let mut selected: HashSet<usize> = HashSet::new();

    for (idx, mp) in map.map_points().iter().enumerate() {
        if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
            continue;
        }
        if mp.found_ratio() < MIN_FOUND_RATIO {
            selected.insert(idx);
            continue;
        }

        // Behind-camera check: ONLY the reference keyframe (ORB-SLAM3 behavior).
        // A point observed by multiple keyframes should not be culled just because
        // it's behind the camera in a non-reference view.
        if let Some(ref_kf) = map.get_keyframe(mp.keyframe_idx) {
            let p_cam = ref_kf.frame.pose_world_to_cam.transform_point(&mp.position);
            if p_cam.z <= 1e-8 {
                selected.insert(idx);
            }
        }
    }

    let mut selected: Vec<usize> = selected.into_iter().collect();
    selected.sort_unstable();
    selected
}

/// Culls landmarks that fail the policy. Returns how many were retired.
///
/// Held under exclusive access so selection and removal see one state.
pub fn cull_landmarks(map: &mut Map) -> usize {
    let mut retired = 0usize;
    for idx in select_for_culling(map) {
        if map.remove_landmark(idx).unwrap_or(false) {
            retired += 1;
        }
    }
    retired
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::mapping::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    #[test]
    fn culls_low_found_ratio_and_clears_its_associations() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]; 2])))
            .unwrap();
        let seed = |feature| LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: feature,
            },
        };
        let doomed = map.insert_landmark(seed(0)).unwrap();
        let kept = map.insert_landmark(seed(1)).unwrap();

        map.set_tracking_stats_for_test(doomed, 10, 1);
        map.set_tracking_stats_for_test(kept, 10, 5);

        assert_eq!(cull_landmarks(&mut map), 1);
        assert!(map.map_points()[doomed].culled);
        assert!(!map.map_points()[kept].culled);
        // Removal cleared the association; no second cleanup pass needed.
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(kept));
        // Retirement is logical: the landmark keeps its records.
        assert!(map.map_points()[doomed].is_observed_by(0));
    }

    #[test]
    fn spares_a_landmark_with_too_few_observations_to_judge() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();
        let idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();
        map.set_tracking_stats_for_test(idx, 4, 0);

        assert_eq!(cull_landmarks(&mut map), 0);
        assert!(!map.map_points()[idx].culled);
    }

    /// Regression test for bug where a landmark observed by multiple keyframes
    /// was incorrectly culled if it was behind the camera in ANY keyframe,
    /// instead of only checking the reference keyframe (mp.keyframe_idx).
    ///
    /// A landmark with reference KF0 at z=5.0 (in front) but also observed by
    /// KF1 rotated 180° (point now behind) should NOT be culled.
    #[test]
    fn behind_camera_culling_only_checks_reference_keyframe() {
        let mut map = Map::new();

        // Reference keyframe (idx=0) - identity pose, point at z=5 is IN FRONT
        map.insert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])))
            .unwrap();

        // Create landmark referenced to KF0
        let mp_idx = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: 0,
                },
            })
            .unwrap();

        // Second keyframe (idx=1) - rotated 180° around Y, point is now BEHIND
        map.insert_keyframe(Keyframe::from_frame(test_frame(1, vec![[0u8; 32]])))
            .unwrap();
        let rotated_rot = SO3F64::exp(Vec3F64::new(0.0, std::f64::consts::PI, 0.0)).matrix();
        let rotated_pose = Pose3d::new(rotated_rot, Vec3F64::ZERO);
        map.set_keyframe_pose_for_test(1, rotated_pose);

        // Link the landmark to KF1 as well (multi-keyframe observation)
        map.link_observation(1, 0, mp_idx).unwrap();

        // Point has good tracking stats (10 visible, 8 found = 80% ratio)
        map.set_tracking_stats_for_test(mp_idx, 10, 8);

        // BUG: Before fix, this would cull the landmark because it's behind KF1
        // FIX: Should NOT cull because reference keyframe (KF0) has point in front
        let culled = cull_landmarks(&mut map);

        assert_eq!(
            culled, 0,
            "Should NOT cull - reference keyframe (0) has point in front"
        );
        assert!(!map.map_points()[mp_idx].culled);
    }
}
