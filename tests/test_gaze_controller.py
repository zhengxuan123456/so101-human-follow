import math
import unittest

import numpy as np

from so101_human_follow.gaze_controller import (
    GazeCalibration, GazeCalibrator, ImageGazeController, PITCH, YAW,
)


SHOULDER = "shoulder_lift.pos"
ELBOW = "elbow_flex.pos"
ROLL = "wrist_roll.pos"


def positions():
    return {YAW: 0.0, PITCH: 0.0, SHOULDER: 0.0, ELBOW: 0.0, ROLL: 0.0, "gripper.pos": 50.0}


class GazeCalibrationTest(unittest.TestCase):
    def test_serialization_preserves_matrix_reference_and_mount_metadata(self):
        calibration = GazeCalibration([[-0.02, 0.006], [0.005, -0.025]], positions(), {"camera": "wrist", "robot": 7})
        restored = GazeCalibration.from_dict(calibration.to_dict())
        np.testing.assert_allclose(restored.jacobian, calibration.jacobian)
        self.assertEqual(restored.reference_positions, calibration.reference_positions)
        self.assertEqual(restored.metadata, {"camera": "wrist", "robot": 7})
        with self.assertRaises(ValueError):
            restored.jacobian[0, 0] = 1.0

    def test_unobservable_implausible_and_nonfinite_responses_are_rejected(self):
        for matrix in (
            [[0.01, 0.02], [0.01, 0.02]], [[0.0001, 0], [0, 0.0001]],
            [[0.1, 0], [0, 0.001]], [[0.3, 0], [0, 0.3]],
            [[math.nan, 0], [0, 0.01]], [[math.inf, 0], [0, 0.01]], [[0.01]],
        ):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                GazeCalibration(matrix, positions())

    def test_reference_and_metadata_must_be_valid(self):
        for reference, metadata in (({}, {}), ({**positions(), ROLL: math.inf}, {}), (positions(), {"x": math.nan})):
            with self.assertRaises(ValueError):
                GazeCalibration(np.eye(2) * 0.02, reference, metadata)
        for value in ({}, {"version": 2}, {"version": 1, "jacobian": None}):
            with self.assertRaises(ValueError):
                GazeCalibration.from_dict(value)


class ImageGazeControllerTest(unittest.TestCase):
    def test_closed_loop_converges_with_reverse_and_rotated_coupled_axes(self):
        for matrix in (
            np.array([[0.02, 0], [0, 0.025]]),
            np.array([[-0.02, 0], [0, -0.025]]),
            np.array([[-0.014, 0.018], [0.017, 0.012]]),
        ):
            with self.subTest(matrix=matrix):
                controller = ImageGazeController(GazeCalibration(matrix, positions()))
                measured = positions()
                initial = np.array([0.23, 0.77])
                for tick in range(130):
                    center = initial + matrix @ np.array([measured[YAW], measured[PITCH]])
                    result = controller.update(center, measured, tick * 0.05, tick * 0.05)
                    self.assertNotIn(result.state, {"fault", "invalid"})
                    self.assertLessEqual(abs(result.delta[0]), 0.5 + 1e-12)
                    self.assertLessEqual(abs(result.delta[1]), 0.4 + 1e-12)
                    measured[YAW] += result.delta[0]
                    measured[PITCH] += result.delta[1]
                self.assertLess(np.linalg.norm(center - 0.5), 0.061)
                self.assertLess(np.linalg.norm(center - 0.5), np.linalg.norm(initial - 0.5) / 5)

    def test_new_frame_time_sets_speed_and_duplicate_frames_do_not_move(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.01, positions()))
        self.assertEqual(controller.update((0.9, 0.9), positions(), 1.0, 1.0).delta, (0, 0))
        result = controller.update((0.9, 0.9), positions(), 1.02, 1.02)
        self.assertLessEqual(abs(result.delta[0]), 10 * 0.02 + 1e-12)
        self.assertLessEqual(abs(result.delta[1]), 8 * 0.02 + 1e-12)
        for now in (1.021, 1.03, 1.05):
            repeated = controller.update((0.9, 0.9), positions(), 1.02, now)
            self.assertEqual(repeated.delta, (0, 0))
        delayed = controller.update((0.9, 0.9), positions(), 1.22, 1.22)
        self.assertLessEqual(max(map(abs, delayed.delta)), 0.5)

    def test_deadband_is_symmetric_and_exposes_current_error(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        controller.update((0.5, 0.5), positions(), 0, 0)
        result = controller.update((0.53, 0.47), positions(), 0.05, 0.05)
        self.assertEqual(result.state, "centered")
        self.assertEqual(result.delta, (0, 0))
        np.testing.assert_allclose(result.error, (0.03, -0.03))

    def test_lost_stale_future_and_changed_targets_hold_and_reconfirm(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        controller.update((0.8, 0.8), positions(), 1, 1, "a")
        self.assertEqual(controller.update(None, positions(), 1.05, 1.05, "a").state, "target_lost")
        self.assertEqual(controller.update((0.8, 0.8), positions(), 1.1, 1.1, "a").delta, (0, 0))
        self.assertEqual(controller.update((0.8, 0.8), positions(), 1.15, 1.5, "a").state, "stale")
        self.assertEqual(controller.update((0.8, 0.8), positions(), 2.0, 1.5, "a").state, "stale")
        result = controller.update((0.8, 0.8), positions(), 1.55, 1.55, "b")
        self.assertEqual(result.state, "target_changed")
        self.assertEqual(result.delta, (0, 0))

    def test_large_geometry_change_invalidates_until_explicit_reset(self):
        for joint, movement in ((SHOULDER, 25.1), (ELBOW, 30.1), (ROLL, 8.1)):
            controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
            result = controller.update((0.8, 0.8), {**positions(), joint: movement}, 1, 1)
            self.assertEqual(result.state, "invalid")
            controller.reset_feedback()
            self.assertEqual(controller.update((0.8, 0.8), positions(), 2, 2).state, "invalid")
            controller.reset_feedback(clear_fault=True)
            self.assertEqual(controller.update((0.8, 0.8), positions(), 3, 3).state, "waiting_frame")

    def test_wrong_direction_faults_after_measured_divergence_even_with_duplicate_frames(self):
        matrix = np.eye(2) * 0.02
        controller = ImageGazeController(GazeCalibration(matrix, positions()))
        measured = positions()
        result = None
        for tick in range(20):
            center = np.array([0.72, 0.72]) - matrix @ np.array([measured[YAW], measured[PITCH]])
            result = controller.update(center, measured, tick * 0.05, tick * 0.05)
            if result.state == "fault":
                break
            measured[YAW] += result.delta[0]
            measured[PITCH] += result.delta[1]
            repeat = controller.update(center, measured, tick * 0.05, tick * 0.05 + 0.01)
            self.assertEqual(repeat.delta, (0, 0))
        self.assertEqual(result.state, "fault")
        self.assertEqual(result.delta, (0, 0))
        self.assertEqual(controller.update((0.5, 0.5), positions(), 2, 2).state, "fault")

    def test_error_growth_without_measured_motion_does_not_claim_direction_fault(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        for tick in range(15):
            result = controller.update((0.6 + 0.01 * tick, 0.6), positions(), tick * 0.05, tick * 0.05)
            self.assertNotEqual(result.state, "fault")

    def test_static_encoders_and_unimproved_error_latch_stall_after_two_seconds(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        for tick in range(65):
            result = controller.update((0.75, 0.65), positions(), tick * 0.05, tick * 0.05, "face")
            if result.state == "fault":
                break
            if tick < 40:
                self.assertIsNone(result.fault_code)
        self.assertEqual(result.fault_code, "stalled")
        self.assertGreaterEqual(tick * 0.05, 2.0)
        self.assertLess(tick * 0.05, 2.3)
        self.assertEqual(result.delta, (0, 0))
        controller.reset_feedback()
        self.assertEqual(controller.update((0.5, 0.5), positions(), 4, 4, "face").fault_code, "stalled")
        controller.reset_feedback(clear_fault=True)
        self.assertEqual(controller.update((0.5, 0.5), positions(), 5, 5, "face").state, "waiting_frame")

    def test_moving_encoders_with_fixed_image_latch_nonconvergence_after_three_seconds(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        measured = positions()
        for tick in range(90):
            result = controller.update((0.75, 0.65), measured, tick * 0.05, tick * 0.05, "face")
            if result.state == "fault":
                break
            measured[YAW] += result.delta[0]
            measured[PITCH] += result.delta[1]
            repeated = controller.update((0.75, 0.65), measured, tick * 0.05, tick * 0.05 + 0.01, "face")
            self.assertEqual(repeated.delta, (0, 0))
        self.assertEqual(result.fault_code, "not_converging")
        self.assertGreaterEqual(tick * 0.05, 3.0)
        self.assertLess(tick * 0.05, 3.3)

    def test_progress_windows_restart_on_target_change_loss_long_gap_or_reset(self):
        controller = ImageGazeController(GazeCalibration(np.eye(2) * 0.02, positions()))
        clock = 0.0
        identity = "face-1"
        for interruption in ("identity", "loss", "gap", "reset"):
            for _ in range(30):
                clock += 0.05
                result = controller.update((0.75, 0.65), positions(), clock, clock, identity)
                self.assertNotEqual(result.state, "fault")
            clock += 0.05
            if interruption == "identity":
                identity = "face-2"
                self.assertEqual(controller.update((0.75, 0.65), positions(), clock, clock, identity).state, "target_changed")
            elif interruption == "loss":
                self.assertEqual(controller.update(None, positions(), clock, clock, identity).state, "target_lost")
            elif interruption == "gap":
                clock += 1.0
                self.assertNotEqual(controller.update((0.75, 0.65), positions(), clock, clock, identity).state, "fault")
            else:
                controller.reset_feedback()

    def test_slow_measured_progress_and_small_deadband_residual_do_not_fault(self):
        matrix = np.eye(2) * 0.02
        controller = ImageGazeController(GazeCalibration(matrix, positions()))
        measured = positions()
        for tick in range(180):
            center = np.array([0.75, 0.65]) + matrix @ np.array([measured[YAW], measured[PITCH]])
            result = controller.update(center, measured, tick * 0.05, tick * 0.05, "face")
            self.assertNotEqual(result.state, "fault")
            measured[YAW] += result.delta[0] * 0.1
            measured[PITCH] += result.delta[1] * 0.1
        self.assertLess(np.linalg.norm(center - 0.5), 0.2)
        controller.reset_feedback()
        for tick in range(180):
            result = controller.update((0.56, 0.51), positions(), 10 + tick * 0.05, 10 + tick * 0.05, "face")
            self.assertNotEqual(result.state, "fault")


class GazeCalibratorTest(unittest.TestCase):
    def setUp(self):
        self.bounds = {YAW: (-20.0, 20.0), PITCH: (-24.0, 24.0)}

    def run_calibration(self, matrix, initial=None, max_step=0.2):
        measured = positions() if initial is None else dict(initial)
        origin = dict(measured)
        calibrator = GazeCalibrator(self.bounds, metadata={"camera": "mount-v1"}, max_step=max_step)
        result = calibrator.begin(measured, (0.5, 0.5), 0, 0, "head-1")
        self.assertEqual(result.targets, {})
        previous_targets = {YAW: measured[YAW], PITCH: measured[PITCH]}
        states = []
        for tick in range(1, 250):
            center = np.array([0.5, 0.5]) + matrix @ np.array([measured[YAW] - origin[YAW], measured[PITCH] - origin[PITCH]])
            result = calibrator.update(measured, center, tick * 0.05, tick * 0.05, "head-1")
            states.append(result.state)
            self.assertTrue(set(result.targets) <= {YAW, PITCH})
            self.assertLessEqual(len(result.targets), 1)
            if result.state in {"baseline", "baseline_pitch"}:
                self.assertEqual(result.targets, {})
            if result.state in {"probe_yaw", "return_yaw"}:
                self.assertTrue(set(result.targets) <= {YAW})
            if result.state in {"probe_pitch", "return_pitch"}:
                self.assertTrue(set(result.targets) <= {PITCH})
            self.assertGreaterEqual(result.progress, 0)
            self.assertLessEqual(result.progress, 1)
            for joint, target in result.targets.items():
                self.assertLessEqual(abs(target - previous_targets[joint]), min(0.2, max_step) + 1e-10)
                low, high = self.bounds[joint]
                self.assertGreaterEqual(target, low)
                self.assertLessEqual(target, high)
                if origin[joint] in (low, high):
                    self.assertNotEqual(target, origin[joint])
                self.assertLessEqual(abs(target - origin[joint]), 2.0 + 1e-10)
                previous_targets[joint] = target
                measured[joint] = target
            if result.state in {"complete", "aborted"}:
                break
        self.assertEqual(measured[SHOULDER], origin[SHOULDER])
        self.assertEqual(measured[ELBOW], origin[ELBOW])
        self.assertEqual(measured[ROLL], origin[ROLL])
        self.assertEqual(measured["gripper.pos"], origin["gripper.pos"])
        return result, measured, states

    def test_measures_coupled_reverse_jacobian_with_bounded_gaze_only_probes(self):
        matrix = np.array([[-0.02, 0.01], [0.012, 0.023]])
        result, measured, states = self.run_calibration(matrix)
        self.assertEqual(result.state, "complete", result.reason)
        np.testing.assert_allclose(result.calibration.jacobian, matrix, atol=1e-10)
        self.assertEqual(result.calibration.metadata, {"camera": "mount-v1"})
        self.assertEqual(result.progress, 1.0)
        self.assertEqual(result.targets, {})
        self.assertLessEqual(abs(measured[YAW]), 0.25)
        self.assertLessEqual(abs(measured[PITCH]), 0.25)
        self.assertIn("baseline_pitch", states)

    def test_uses_negative_probe_when_positive_side_has_less_room(self):
        initial = {**positions(), YAW: 18.0, PITCH: 22.0}
        result, measured, _ = self.run_calibration(np.eye(2) * 0.02, initial)
        self.assertEqual(result.state, "complete", result.reason)
        np.testing.assert_allclose(result.calibration.jacobian, np.eye(2) * 0.02)

    def test_recorded_endpoints_probe_one_axis_at_a_time_and_finish_inside(self):
        matrix = np.array([[-0.02, 0.01], [0.012, 0.023]])
        for yaw, pitch in ((-20.0, -24.0), (20.0, 24.0), (-20.0, 24.0), (20.0, -24.0)):
            with self.subTest(yaw=yaw, pitch=pitch):
                result, measured, _ = self.run_calibration(matrix, {**positions(), YAW: yaw, PITCH: pitch})
                self.assertEqual(result.state, "complete", result.reason)
                np.testing.assert_allclose(result.calibration.jacobian, matrix, atol=1e-10)
                expected_yaw = yaw + (0.5 if yaw < 0 else -0.5)
                expected_pitch = pitch + (0.5 if pitch < 0 else -0.5)
                self.assertAlmostEqual(measured[YAW], expected_yaw)
                self.assertAlmostEqual(measured[PITCH], expected_pitch)
                self.assertAlmostEqual(result.calibration.reference_positions[YAW], expected_yaw)
                self.assertAlmostEqual(result.calibration.reference_positions[PITCH], expected_pitch)

    def test_both_normalized_minus100_start_without_baseline_hold(self):
        self.bounds = {YAW: (-100.0, 100.0), PITCH: (-100.0, 100.0)}
        initial = {**positions(), YAW: -100.0, PITCH: -100.0}
        calibrator = GazeCalibrator(self.bounds)
        result = calibrator.begin(initial, (0.5, 0.5), 0, 0, "head")
        self.assertEqual(result.state, "baseline")
        self.assertEqual(result.targets, {})
        for tick in range(1, 4):
            result = calibrator.update(initial, (0.5, 0.5), tick * 0.05, tick * 0.05, "head")
            self.assertEqual(result.state, "baseline")
            self.assertEqual(result.targets, {})
        matrix = np.array([[0.02, 0.008], [-0.006, -0.025]])
        result, measured, _ = self.run_calibration(matrix, initial)
        self.assertEqual(result.state, "complete", result.reason)
        np.testing.assert_allclose(result.calibration.jacobian, matrix, atol=1e-10)
        self.assertAlmostEqual(measured[YAW], -99.5)
        self.assertAlmostEqual(measured[PITCH], -99.5)

    def test_return_face_is_compared_to_predicted_interior_endpoint(self):
        # A half-unit interior rest shifts the face by 0.05, above the old
        # 0.025 return threshold. This valid geometric residual must be retained.
        matrix = np.array([[0.1, 0.0], [0.0, -0.1]])
        result, measured, _ = self.run_calibration(matrix, {**positions(), YAW: -20.0, PITCH: 24.0})
        self.assertEqual(result.state, "complete", result.reason)
        np.testing.assert_allclose(result.calibration.jacobian, matrix, atol=1e-10)

    def test_respects_a_smaller_configured_step_cap_without_expanding_probe(self):
        result, _, _ = self.run_calibration(np.eye(2) * 0.02, max_step=0.07)
        self.assertEqual(result.state, "complete", result.reason)
        for invalid in (0.0, -0.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                GazeCalibrator(self.bounds, max_step=invalid)

    def test_small_head_drift_on_return_invalidates_measurement(self):
        measured = positions()
        calibrator = GazeCalibrator(self.bounds)
        calibrator.begin(measured, (0.5, 0.5), 0, 0, "head")
        shifted = False
        for tick in range(1, 180):
            if calibrator.state == "return_yaw":
                shifted = True
            center = np.array([0.5, 0.5]) + np.eye(2) * 0.02 @ np.array([measured[YAW], measured[PITCH]])
            if shifted:
                center[0] += 0.03
            result = calibrator.update(measured, center, tick * 0.05, tick * 0.05, "head")
            measured.update(result.targets)
            if result.state in {"complete", "aborted"}:
                break
        self.assertEqual(result.state, "aborted")
        self.assertIn("Head moved", result.reason)
        self.assertEqual(result.targets, {})

    def test_unobservable_response_is_rejected_after_safe_return(self):
        for matrix in (np.zeros((2, 2)), np.array([[0.02, 0.02], [0.01, 0.01]])):
            result, _, _ = self.run_calibration(matrix)
            self.assertEqual(result.state, "aborted")
            self.assertIsNone(result.calibration)
            self.assertEqual(result.targets, {})

    def test_missing_stale_identity_clock_and_body_motion_abort_without_return_command(self):
        for case in ("missing", "stale", "identity", "body", "time"):
            with self.subTest(case=case):
                calibrator = GazeCalibrator(self.bounds)
                measured = positions()
                calibrator.begin(measured, (0.5, 0.5), 1, 1, "a")
                for tick in range(1, 5):
                    result = calibrator.update(measured, (0.5, 0.5), 1 + tick * 0.05, 1 + tick * 0.05, "a")
                    measured.update(result.targets)
                point, frame, now, identity = (0.5, 0.5), 1.25, 1.25, "a"
                if case == "missing":
                    point = None
                elif case == "stale":
                    now += 1
                elif case == "identity":
                    identity = "b"
                elif case == "body":
                    measured[ROLL] += 0.51
                else:
                    frame = now = 22
                result = calibrator.update(measured, point, frame, now, identity)
                self.assertEqual(result.state, "aborted")
                self.assertEqual(result.targets, {})
                self.assertEqual(calibrator.update(positions(), (0.5, 0.5), 23, 23, "a").targets, {})

    def test_duplicate_frames_do_not_count_as_samples_or_issue_motion(self):
        calibrator = GazeCalibrator(self.bounds)
        calibrator.begin(positions(), (0.5, 0.5), 1, 1)
        for tick in range(1, 5):
            result = calibrator.update(positions(), (0.5, 0.5), 1, 1 + tick * 0.02)
            self.assertEqual(result.state, "baseline")
            self.assertEqual(result.targets, {})
        for tick in (1, 2):
            result = calibrator.update(positions(), (0.5, 0.5), 1 + tick * 0.05, 1 + tick * 0.05)
            self.assertEqual(result.state, "baseline")
        result = calibrator.update(positions(), (0.5, 0.5), 1.15, 1.15)
        self.assertEqual(result.state, "baseline")
        result = calibrator.update(positions(), (0.5, 0.5), 1.2, 1.2)
        self.assertEqual(result.state, "probe_yaw")

    def test_start_outside_bounds_or_without_head_cannot_issue_a_probe(self):
        for measured, point in (({**positions(), YAW: 20.01}, (0.5, 0.5)), (positions(), None)):
            calibrator = GazeCalibrator(self.bounds)
            result = calibrator.begin(measured, point, 1, 1)
            self.assertEqual(result.state, "aborted")
            self.assertEqual(result.targets, {})

    def test_measured_feedback_outside_bounds_aborts_without_recovery_target(self):
        initial = {**positions(), YAW: -20.0, PITCH: -24.0}
        calibrator = GazeCalibrator(self.bounds)
        calibrator.begin(initial, (0.5, 0.5), 0, 0, "head")
        result = calibrator.update({**initial, PITCH: -24.01}, (0.5, 0.5), 0.05, 0.05, "head")
        self.assertEqual(result.state, "aborted")
        self.assertEqual(result.targets, {})

    def test_sampling_waits_for_actual_motor_arrival_and_cancel_never_returns(self):
        calibrator = GazeCalibrator(self.bounds)
        calibrator.begin(positions(), (0.5, 0.5), 0, 0)
        for tick in range(1, 60):
            result = calibrator.update(positions(), (0.5, 0.5), tick * 0.05, tick * 0.05)
        self.assertEqual(result.state, "probe_yaw")
        result = calibrator.cancel("operator STOP")
        self.assertEqual(result.reason, "operator STOP")
        self.assertEqual(result.targets, {})


if __name__ == "__main__":
    unittest.main()
