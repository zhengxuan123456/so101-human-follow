"""Exercise the real engine/routes with a camera/encoder plant, never hardware."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from so101_human_follow.config import load_config
from so101_human_follow.gaze_controller import GazeCalibration, ImageGazeController
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.robot_adapter import CalibrationRangeError, preview_initial_positions
from so101_human_follow.web_app import create_app
from so101_human_follow.web_engine import ControlCommand, ControlEngine, ControlError

YAW, PITCH = "shoulder_pan.pos", "wrist_flex.pos"


class GazeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.clock = 20.0
        self.timer = patch("so101_human_follow.web_engine.time.monotonic", side_effect=lambda: self.clock)
        self.timer.start()
        self.addCleanup(self.timer.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile = Path(self.directory.name) / "safety_profile.json"
        self.config = load_config("config.example.json")
        self.gaze_path = self.profile.with_name("gaze_calibration.json")
        self.engine = ControlEngine(self.config, robot_allowed=True, profile_path=self.profile, gaze_path=self.gaze_path)
        self.client_id = "gaze_integration_operator"
        self.positions = preview_initial_positions()
        self.writes, self.holds = [], 0
        # An inclined camera: both joints affect both image coordinates; pitch
        # has the opposite sign to the original hardcoded positive correction.
        self.matrix = np.array([[-0.025, 0.007], [0.006, 0.022]])
        self.anchor = np.array([0.67, 0.65])
        self.target_id = "face-1"

        def hold():
            self.holds += 1
            return dict(self.positions)

        def raw_observation():
            raw = {joint: (value * 20.0 if joint == "gripper.pos" else (value + 100.0) * 10.0)
                   for joint, value in self.positions.items()}
            return {"positions": {joint: min(100.0, max(0.0 if joint == "gripper.pos" else -100.0, value))
                                   for joint, value in self.positions.items()},
                    "raw_positions": raw, "raw_limits": {joint: (0, 2000) for joint in raw},
                    "clipped_joints": tuple(joint for joint, value in raw.items() if not 0 <= value <= 2000)}

        def send(targets, active, fixed, *, before_write, verify_calibrated_raw=False, raw_guard_joints=()):
            self.assertTrue(verify_calibrated_raw)
            observation = raw_observation()
            observed = observation["positions"]
            clipped = tuple(joint for joint in set((*active, *fixed, *raw_guard_joints))
                            if joint in observation["clipped_joints"])
            if clipped:
                raise CalibrationRangeError(observation, clipped)
            if before_write(observed):
                self.writes.append({joint: float(targets[joint]) for joint in active})
                self.positions.update(self.writes[-1])
            return dict(self.positions), observed

        self.engine._robot = SimpleNamespace(
            connected=True, read_positions=lambda: dict(self.positions),
            read_calibration_observation=raw_observation,
            send_agent_targets=send, hold_current=hold, hold_raw_current=hold,
        )
        self.engine._robot_state = "CONNECTED_DISARMED"
        self.engine._calibration_fingerprint = "test-encoder-calibration"
        self.engine._safe_profile = SimpleNamespace(
            controlled_limits=dict(self.config.safety.absolute_limits), fixed_positions={},
        )
        self.engine._companion_mechanical_limits = {
            joint: ((0.0, 100.0) if joint == "gripper.pos" else (-100.0, 100.0)) for joint in self.positions
        }
        self.engine.heartbeat(self.client_id)
        self.engine._camera_owner = self.client_id
        self.engine._camera_name = self.config.camera.wrist_name
        self.engine._preview_state = "RUNNING"
        self.frame()

    def frame(self, *, missing=False):
        # Include nominal pitch compensation when expressive joints move.
        pitch = (self.positions[PITCH] + 1.05 * self.positions["shoulder_lift.pos"]
                 + 1.02 * self.positions["elbow_flex.pos"])
        center = self.anchor + self.matrix @ [self.positions[YAW], pitch]
        x, y = map(float, center)
        detection = PoseDetection(None, (), 0, "no person") if missing else PoseDetection(
            None, (), 0, "no person", head_center=(x, y), head_box=(x - 0.06, y - 0.08, x + 0.06, y + 0.08),
            head_confidence=0.85, head_source="face", head_verified=True, head_state="verified",
            head_track_id=self.target_id,
        )
        self.engine._update_tracking(detection, self.clock, captured_at=self.clock - 0.01)
        return center

    def tick(self, *, missing=False):
        self.clock += 0.06
        self.engine.heartbeat(self.client_id)
        self.frame(missing=missing)
        self.engine._hold_agent_if_pending()
        self.engine._poll_gaze_calibration(self.clock)
        self.engine._poll_companion(self.clock)

    def start(self):
        return self.engine.request_gaze_calibration(
            client_id=self.client_id, action="start", expected_epoch=self.engine._epoch, confirmation="123",
        )

    def install_calibration(self):
        calibration = GazeCalibration(self.matrix, self.positions, self.engine._gaze_metadata_locked())
        self.engine._gaze_calibration = calibration
        self.engine._gaze_controller = ImageGazeController(calibration)
        self.engine._gaze_state = "ready"
        return calibration

    def wait_until_probing(self):
        for _ in range(90):
            self.tick()
            if self.engine._gaze_calibrator is not None:
                return
        self.fail(self.engine._gaze_reason)

    def finish_calibration(self):
        self.start()
        for _ in range(300):
            self.tick()
            if self.engine._gaze_client is None:
                break
        self.assertEqual(self.engine._gaze_state, "ready", self.engine._gaze_reason)

    def test_explicit_calibration_measures_coupled_reverse_directions_and_persists(self):
        self.start()
        self.assertEqual(self.writes, [])  # HTTP request never performs servo IO.
        previous = dict(self.positions)
        for _ in range(300):
            self.tick()
            for joint in (YAW, PITCH):
                self.assertLessEqual(abs(self.positions[joint] - previous[joint]), 0.200001)
                self.assertLessEqual(abs(self.positions[joint]), 2.000001)
            previous = dict(self.positions)
            if self.engine._gaze_client is None:
                break
        self.assertEqual(self.engine._gaze_state, "ready", self.engine._gaze_reason)
        self.assertTrue(self.writes)
        self.assertTrue(all(len(write) == 1 and set(write) <= {YAW, PITCH} for write in self.writes))
        np.testing.assert_allclose(self.engine._gaze_calibration.jacobian, self.matrix, atol=1e-8)
        stored = json.loads((self.profile.parent / "gaze_calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["metadata"]["calibration_fingerprint"], "test-encoder-calibration")
        loaded = ControlEngine(self.config, robot_allowed=True, profile_path=self.profile, gaze_path=self.gaze_path)
        np.testing.assert_allclose(loaded._gaze_calibration.jacobian, self.matrix)
        self.assertIsNone(loaded._gaze_client)
        self.assertIsNone(loaded._companion_client)

    def test_calibrated_companion_centers_a_coupled_camera_with_body_motion(self):
        self.finish_calibration()
        self.tick()  # finish the pending hold first
        initial_error = float(np.linalg.norm(self.frame() - 0.5))
        self.engine.set_companion(client_id=self.client_id, enabled=True, allow_unoccupied_idle=False,
                                  expected_epoch=self.engine._epoch)
        for _ in range(190):
            self.tick()
        final_error = float(np.linalg.norm(self.frame() - 0.5))
        self.assertLess(final_error, initial_error * 0.45, self.engine._gaze_reason)
        self.assertTrue(self.engine._gaze_ready_locked(), self.engine._gaze_reason)

    def test_loss_or_target_change_aborts_without_blind_return(self):
        for change in ("loss", "identity"):
            with self.subTest(change=change):
                self.target_id = "face-1"
                self.frame()
                self.start()
                self.wait_until_probing()
                for _ in range(8):
                    self.tick()
                before = len(self.writes)
                if change == "identity":
                    self.target_id = "face-2"
                self.tick(missing=change == "loss")
                self.assertIsNone(self.engine._gaze_client)
                self.assertEqual(len(self.writes), before)
                self.assertFalse(self.engine._gaze_ready_locked())
                self.engine._hold_agent_if_pending()

    def test_stop_event_is_not_cleared_by_calibration_tick(self):
        self.start()
        self.wait_until_probing()
        for _ in range(8):
            self.tick()
        before = len(self.writes)
        self.engine._motion_stop.set()  # STOP may be waiting for the send lock.
        self.tick()
        self.assertIsNone(self.engine._gaze_client)
        self.assertEqual(len(self.writes), before)

    def test_operator_stop_and_lease_loss_cancel_calibration(self):
        self.start()
        self.wait_until_probing()
        self.engine.disarm("test STOP")
        self.assertIsNone(self.engine._gaze_client)
        self.assertTrue(self.engine._motion_stop.is_set())
        self.engine._hold_agent_if_pending()
        self.frame()
        self.start()
        self.tick()
        self.clock += 2.1
        self.engine._lease_watchdog(self.clock)
        self.assertIsNone(self.engine._gaze_client)

    def test_recalibration_invalidates_old_disk_data_before_probing(self):
        calibration = self.install_calibration()
        self.engine._save_gaze_calibration(calibration)
        self.start()
        self.wait_until_probing()
        self.engine.request_gaze_calibration(client_id=self.client_id, action="cancel", expected_epoch=None)
        loaded = ControlEngine(self.config, robot_allowed=True, profile_path=self.profile, gaze_path=self.gaze_path)
        self.assertIsNone(loaded._gaze_calibration)

    def test_camera_changes_or_fault_require_recalibration(self):
        self.install_calibration()
        self.assertTrue(self.engine._gaze_ready_locked())
        self.engine._camera_name = "another physical camera"
        self.assertFalse(self.engine._gaze_ready_locked())
        self.engine._camera_name = self.config.camera.wrist_name
        self.engine._gaze_state = "fault"
        with self.engine._frame_condition:
            self.engine._clear_camera_tracking_locked("camera disconnected")
        self.assertEqual(self.engine._gaze_state, "fault")
        self.assertFalse(self.engine._gaze_ready_locked())

    def test_uncalibrated_companion_never_guesses_motor_signs(self):
        with self.assertRaises(ControlError):
            self.engine.set_companion(client_id=self.client_id, enabled=True, allow_unoccupied_idle=True,
                                      expected_epoch=self.engine._epoch)
        self.assertEqual(self.writes, [])

    def test_other_motor_commands_are_blocked_during_calibration(self):
        self.start()
        with self.assertRaises(ControlError):
            self.engine._execute_command(ControlCommand("agent_plan_start", {}, self.client_id))
        self.assertEqual(self.writes, [])

    def test_api_validation_and_cancel_invalidates_older_start_epoch(self):
        app = create_app(self.engine, control_token="gaze-test-token")
        app.config.update(TESTING=True)
        client = app.test_client()
        client.get("/")
        headers = {"X-Client-Id": self.client_id, "Origin": "http://localhost"}
        old_epoch = self.engine._epoch
        for payload in ([], {}, {"action": []}, {"action": "start", "expected_epoch": True}):
            response = client.post("/api/gaze/calibrate", json=payload, headers=headers)
            self.assertEqual(response.status_code, 400)
        response = client.post("/api/gaze/calibrate", json={"action": "cancel"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        response = client.post("/api/gaze/calibrate", json={"action": "start", "expected_epoch": old_epoch,
                               "confirmation": "123"}, headers=headers)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.writes, [])

    def test_frame_expiring_during_serial_io_cannot_write_a_head_correction(self):
        self.install_calibration()
        self.engine.set_companion(client_id=self.client_id, enabled=True, allow_unoccupied_idle=False,
                                  expected_epoch=self.engine._epoch)
        original = self.engine._robot.send_agent_targets
        def delayed_send(*args, **kwargs):
            self.clock += 0.26
            return original(*args, **kwargs)
        self.engine._robot.send_agent_targets = delayed_send
        for _ in range(20):
            self.tick()
        self.assertEqual(self.writes, [])

    def test_search_times_out_and_cannot_restart_without_new_target_or_request(self):
        self.install_calibration()
        self.engine.set_companion(client_id=self.client_id, enabled=True, allow_unoccupied_idle=True,
                                  allow_vertical_search=True, expected_epoch=self.engine._epoch)
        for _ in range(150):
            self.tick(missing=True)
        before = len(self.writes)
        self.assertEqual(self.engine._companion_state, "search_timeout")
        for _ in range(15):
            self.tick(missing=True)
        self.assertEqual(len(self.writes), before)

    def test_click_starts_countdown_without_face_or_motor_goals(self):
        self.frame(missing=True)
        result = self.start()
        self.assertTrue(result["gaze"]["calibrating"])
        self.assertTrue(result["gaze"]["preparing"])
        self.assertEqual(result["gaze"]["countdown_remaining"], 3)
        for _ in range(65):
            self.tick(missing=True)
        self.assertEqual(self.engine._gaze_phase, "waiting_face")
        self.assertEqual(self.writes, [])
        self.assertEqual(self.holds, 0)
        for _ in range(30):
            self.tick()
        self.assertFalse(self.engine._gaze_preparing)
        self.assertTrue(self.writes)

    def test_waiting_can_be_cancelled_without_writing_a_hold_or_invalidating_saved_data(self):
        calibration = self.install_calibration()
        self.engine._save_gaze_calibration(calibration)
        self.start()
        for _ in range(5):
            self.tick(missing=True)
        self.engine.request_gaze_calibration(client_id=self.client_id, action="cancel", expected_epoch=None)
        self.engine._hold_agent_if_pending()
        self.assertFalse(self.engine._agent_hold_pending)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.holds, 0)
        self.assertIsNotNone(GazeCalibration.from_dict(json.loads(self.gaze_path.read_text(encoding="utf-8"))))

    def test_chosen_comfortable_pose_can_start_outside_human_mimic_envelope(self):
        self.positions[YAW] = -81.77
        self.anchor -= self.matrix @ [self.positions[YAW], self.positions[PITCH]]
        self.frame()
        self.finish_calibration()
        self.assertAlmostEqual(self.engine._gaze_calibration.reference_positions[YAW], -81.77)
        self.assertEqual(self.engine.agent_joint_limits()[YAW], (-70.0, 70.0))
        self.assertEqual(self.engine.companion_joint_limits()[YAW], (-100.0, 100.0))
        for write in self.writes:
            if YAW in write:
                self.assertTrue(-95.0 <= write[YAW] <= -61.77)

    def test_actual_mechanical_boundary_does_not_send_any_goal_while_waiting(self):
        self.positions[YAW] = -101.0
        self.anchor -= self.matrix @ [self.positions[YAW], self.positions[PITCH]]
        self.frame()
        self.start()
        for _ in range(75):
            self.tick()
        self.assertEqual(self.engine._gaze_phase, "calibration_range_mismatch")
        self.assertIsNone(self.engine._gaze_calibrator)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.positions[YAW], -101.0)
        self.assertEqual(self.holds, 0)

    def test_waiting_allows_owner_connection_but_blocks_other_motion(self):
        robot = self.engine._robot
        self.engine._robot = None
        self.start()
        for _ in range(60):
            self.tick()
        self.assertEqual(self.engine._gaze_phase, "waiting_robot")
        with patch.object(self.engine, "_connect_robot", return_value={"ok": True}) as connect:
            self.engine._execute_command(ControlCommand("connect", {"confirmation": "123"}, self.client_id))
            connect.assert_called_once()
        with self.assertRaises(ControlError):
            self.engine._execute_command(ControlCommand("arm", {}, self.client_id))
        self.engine._robot = robot
        for _ in range(25):
            self.tick()
        self.assertFalse(self.engine._gaze_preparing)

    def test_preparation_timeout_never_starts_powered_motion(self):
        self.start()
        self.clock += 64.0
        self.tick(missing=True)
        self.assertIsNone(self.engine._gaze_client)
        self.assertEqual(self.writes, [])
        self.assertFalse(self.engine._agent_hold_pending)

    def test_switching_camera_during_preparation_resets_detection_wait(self):
        self.start()
        self.engine._gaze_prepare_sample = {"old": True}
        self.engine.disconnect_camera()
        self.assertEqual(self.engine._gaze_client, self.client_id)
        self.assertIsNone(self.engine._gaze_prepare_sample)
        self.assertEqual(self.engine._gaze_phase, "waiting_camera")
        self.assertEqual(self.writes, [])

    def test_both_recorded_endpoints_probe_only_inward_and_finish_slightly_inside(self):
        self.positions[YAW] = self.positions[PITCH] = -100.0
        self.anchor -= self.matrix @ [self.positions[YAW], self.positions[PITCH]]
        self.frame()
        self.finish_calibration()
        self.assertAlmostEqual(self.positions[YAW], -99.5)
        self.assertAlmostEqual(self.positions[PITCH], -99.5)
        reference = self.engine._gaze_calibration.reference_positions
        self.assertAlmostEqual(reference[YAW], -99.5)
        self.assertAlmostEqual(reference[PITCH], -99.5)
        for write in self.writes:
            self.assertEqual(len(write), 1)
            for joint, value in write.items():
                self.assertIn(joint, (YAW, PITCH))
                self.assertTrue(-100.0 < value <= -98.0)
        np.testing.assert_allclose(self.engine._gaze_calibration.jacobian, self.matrix, atol=1e-8)

    def test_clipped_raw_position_is_reported_immediately_without_probe_or_return(self):
        self.positions[PITCH] = -101.0
        self.anchor -= self.matrix @ [self.positions[YAW], self.positions[PITCH]]
        self.frame()
        self.start()
        for _ in range(60):
            self.tick()
        self.assertIsNone(self.engine._gaze_client)
        self.assertEqual(self.engine._gaze_phase, "calibration_range_mismatch")
        self.assertEqual(self.engine._gaze_raw_observation["positions"][PITCH], -100.0)
        self.assertLess(self.engine._gaze_raw_observation["raw_positions"][PITCH], 0)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.positions[PITCH], -101.0)
        self.assertEqual(self.holds, 0)

    def test_raw_read_failure_is_a_terminal_reason_instead_of_waiting_for_a_face(self):
        self.engine._robot.read_calibration_observation = lambda: (_ for _ in ()).throw(OSError("read failed"))
        self.start()
        for _ in range(60):
            self.tick()
        self.assertIsNone(self.engine._gaze_client)
        self.assertEqual(self.engine._gaze_phase, "calibration_read_failed")
        self.assertEqual(self.writes, [])
        self.assertEqual(self.holds, 0)

    def test_raw_drift_during_probe_aborts_without_a_clipped_return_goal(self):
        self.start()
        self.wait_until_probing()
        for _ in range(8):
            self.tick()
        self.assertTrue(self.writes)
        self.positions[PITCH] = -101.0
        self.anchor = np.array([0.6, 0.6]) - self.matrix @ [self.positions[YAW], self.positions[PITCH]]
        before = len(self.writes)
        self.tick()
        self.assertEqual(self.engine._gaze_phase, "calibration_range_mismatch")
        self.assertEqual(len(self.writes), before)
        self.engine._hold_agent_if_pending()
        self.assertEqual(self.positions[PITCH], -101.0)
        self.assertEqual(len(self.writes), before)
        self.assertGreater(self.holds, 0)

    def test_inactive_camera_axis_raw_drift_between_read_and_write_stops_the_probe(self):
        original = self.engine._robot.send_agent_targets
        def drift_before_send(*args, **kwargs):
            self.positions[PITCH] = -101.0
            return original(*args, **kwargs)
        self.engine._robot.send_agent_targets = drift_before_send
        self.start()
        for _ in range(100):
            self.tick()
            if self.engine._gaze_client is None:
                break
        self.assertEqual(self.engine._gaze_phase, "calibration_range_mismatch")
        self.assertEqual(self.writes, [])
        self.assertEqual(self.positions[PITCH], -101.0)


if __name__ == "__main__":
    unittest.main()
