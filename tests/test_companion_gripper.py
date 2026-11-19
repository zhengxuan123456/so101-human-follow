import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from so101_human_follow.agent_actions import VisualProximity
from so101_human_follow.config import load_config
from so101_human_follow.gaze_controller import GazeCalibration, ImageGazeController
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.pose_math import Landmark, PoseFeatures
from so101_human_follow.robot_adapter import preview_initial_positions
from so101_human_follow.web_engine import ControlEngine
from so101_human_follow.web_app import create_app


GRIPPER = "gripper.pos"


class FakeFollower:
    connected = True

    def __init__(self):
        self.positions = preview_initial_positions()
        self.writes = []
        self.fixed_writes = []
        self.holds = 0
        self.read_count = 0

    def read_positions(self):
        self.read_count += 1
        return dict(self.positions)

    def read_calibration_observation(self):
        positions = self.read_positions()
        raw = {joint: value * 20.0 if joint == GRIPPER else (value + 100.0) * 10.0
               for joint, value in positions.items()}
        limits = {joint: (0.0, 2000.0) for joint in positions}
        return {"positions": positions, "raw_positions": raw, "raw_limits": limits,
                "clipped_joints": tuple(joint for joint, value in raw.items()
                                        if not limits[joint][0] <= value <= limits[joint][1])}

    def hold_current(self):
        self.holds += 1
        return self.read_positions()

    hold_raw_current = hold_current

    def send_agent_targets(self, targets, active_joints, fixed_positions, *, before_write=None,
                           verify_calibrated_raw=False):
        assert type(verify_calibrated_raw) is bool
        if verify_calibrated_raw:
            observation = self.read_calibration_observation()
            assert not observation["clipped_joints"], observation["clipped_joints"]
            for joint, value in {**fixed_positions, **{joint: targets[joint] for joint in active_joints}}.items():
                low, high = (0.0, 100.0) if joint == GRIPPER else (-100.0, 100.0)
                assert low <= value <= high, (joint, value)
        observed = self.read_positions()
        if before_write is None or before_write(observed):
            action = {joint: targets[joint] for joint in active_joints}
            self.writes.append(action)
            self.fixed_writes.append(dict(fixed_positions))
            self.positions.update(action)
            self.positions.update(fixed_positions)
        return self.read_positions(), observed


class CompanionGripperTest(unittest.TestCase):
    def setUp(self):
        self.clock = 10.0
        timer = patch("so101_human_follow.web_engine.time.monotonic", side_effect=lambda: self.clock)
        timer.start()
        self.addCleanup(timer.stop)
        config = load_config("config.example.json")
        self.client_id = "operator"
        self.robot = FakeFollower()
        self.engine = ControlEngine(config, robot_allowed=True)
        self.engine._robot = self.robot
        self.engine._robot_state = "CONNECTED_DISARMED"
        self.engine._safe_profile = SimpleNamespace(
            controlled_limits={**config.safety.absolute_limits, GRIPPER: (12.0, 78.0)},
            fixed_positions={},
        )
        self.engine._companion_mechanical_limits = self.engine.agent_joint_limits()
        self.engine.heartbeat(self.client_id)
        self.engine._camera_owner = "operator"
        self.engine._camera_name = config.camera.wrist_name
        self.engine._preview_state = "RUNNING"
        self.engine._companion_frame_at = self.clock
        self.install_gaze_reference()
        self.engine.set_companion(
            client_id="operator", enabled=True, allow_unoccupied_idle=True,
            expected_epoch=self.engine._epoch,
        )

    def install_gaze_reference(self):
        # This suite exercises gripper/handoff behavior with an already measured
        # mounting. Calibration itself is tested with a simulated image plant.
        calibration = GazeCalibration(
            [[-0.02, 0.0], [0.0, -0.02]], self.robot.positions,
            self.engine._gaze_metadata_locked(),
        )
        self.engine._gaze_calibration = calibration
        self.engine._gaze_controller = ImageGazeController(calibration)
        self.engine._gaze_state = "ready"

    def prepare_near_elbow_boundary(self):
        self.robot.positions["elbow_flex.pos"] = -69.5
        self.install_gaze_reference()

    def tick(self, openness, *, face=False, proximity="SAFE"):
        self.clock += 0.06
        self.engine.heartbeat(self.client_id)
        self.engine._companion_frame_at = self.clock
        self.engine._visual_proximity = VisualProximity(proximity, "test frame")
        self.engine._detection = PoseDetection(
            PoseFeatures(0.0, 0.0, 0.0, 0.5, hand_open=openness or 0.0) if openness is not None else None,
            (Landmark(0.5, 0.5),) * 33 if face else (),
            int(face), "test frame",
            hand_landmarks=(Landmark(0.5, 0.5),) if openness is not None else (),
            valid_features=frozenset({"hand_open"}) if openness is not None else frozenset(),
            head_center=(0.5, 0.5) if face else None,
            head_box=(0.4, 0.4, 0.6, 0.6) if face else None,
            head_confidence=0.95 if face else 0.0,
            head_source="face" if face else None,
            head_track_id="test-head" if face else None,
            head_verified=face, head_state="verified" if face else "",
        )
        self.engine._hold_agent_if_pending()
        self.engine._poll_companion(self.clock)

    def test_hand_without_face_moves_only_gripper_inside_calibrated_limits(self):
        for _ in range(20):
            self.tick(0.9)
        self.assertGreater(self.robot.positions[GRIPPER], 70.0)
        for _ in range(25):
            self.tick(0.1)
        self.assertLess(self.robot.positions[GRIPPER], 20.0)
        previous = preview_initial_positions()[GRIPPER]
        for write in self.robot.writes:
            self.assertEqual(set(write), {GRIPPER})
            current = write[GRIPPER]
            self.assertTrue(12.0 < current < 78.0)
            self.assertLessEqual(abs(current - previous), 6.0)
            previous = current

    def test_hand_loss_holds_the_gripper_and_face_tracking_resumes(self):
        for _ in range(20):
            self.tick(0.9, face=True)
        self.assertIn(GRIPPER, self.robot.writes[-1])
        held_gripper = self.robot.positions[GRIPPER]
        writes_before_loss = len(self.robot.writes)
        self.tick(None, face=True)
        self.assertEqual(len(self.robot.writes), writes_before_loss)
        self.assertTrue(self.engine._agent_hold_pending)
        self.tick(None, face=True)
        self.assertEqual(self.robot.holds, 1)
        self.assertEqual(len(self.robot.writes), writes_before_loss)
        self.tick(None, face=True)
        self.assertNotIn(GRIPPER, self.robot.writes[-1])
        self.assertEqual(self.robot.positions[GRIPPER], held_gripper)
        self.assertGreater(len(self.robot.writes), writes_before_loss)

    def test_too_close_blocks_even_a_reliable_hand(self):
        for _ in range(20):
            self.tick(0.9, proximity="TOO_CLOSE")
        self.assertEqual(self.robot.writes, [])
        self.assertEqual(self.engine._companion_state, "too_close")

    def test_boundary_crossing_stops_that_axis_and_resumes_remaining_axes(self):
        self.prepare_near_elbow_boundary()
        for _ in range(20):
            self.tick(None, face=True)
        self.robot.positions["elbow_flex.pos"] = -70.09936766034326
        before = len(self.robot.writes)
        self.tick(None, face=True)
        self.assertEqual(len(self.robot.writes), before)
        self.assertTrue(self.engine._agent_hold_pending)
        self.assertEqual(self.engine._companion_client, "operator")
        self.assertEqual(self.engine._companion_state, "range_blocked")
        for _ in range(20):
            self.tick(None, face=True)
        self.assertGreater(len(self.robot.writes), before)
        for write in self.robot.writes[before:]:
            self.assertNotIn("elbow_flex.pos", write)
        self.assertEqual(self.robot.positions["elbow_flex.pos"], -70.09936766034326)
        self.assertEqual(self.engine._companion_client, "operator")
        self.assertIsNone(self.engine._error)

    def test_agent_motion_takes_over_then_companion_resumes_from_feedback(self):
        self.prepare_near_elbow_boundary()
        for _ in range(20):
            self.tick(None, face=True)
        # A handoff must capture a fresh pose, including a small encoder drift.
        self.robot.positions["elbow_flex.pos"] = -70.189701897
        self.engine._start_agent_plan(
            [{"name": "perform_greeting", "arguments": {"count": 1}}],
            client_id="operator", submitted_epoch=self.engine._epoch,
        )
        self.assertEqual(self.engine._companion_client, "operator")
        self.assertEqual(self.engine._agent_recovery_positions["elbow_flex.pos"], -70.189701897)
        start = len(self.robot.writes)
        last_control = 0.0
        for _ in range(300):
            self.clock += 0.05
            self.engine.heartbeat("operator")
            self.engine._companion_frame_at = self.clock
            last_control = self.engine._send_agent_if_due(self.clock, last_control)
            before_companion = len(self.robot.writes)
            self.engine._poll_companion(self.clock)
            self.assertEqual(len(self.robot.writes), before_companion)
            self.assertEqual(self.engine._companion_client, "operator")
            if self.engine._agent_plan is None:
                break
        self.assertEqual(self.engine._agent_motion_state, "COMPLETE")
        self.assertGreater(len(self.robot.writes), start)
        for targets in self.robot.writes[start:] + self.robot.fixed_writes[start:]:
            self.assertNotIn("elbow_flex.pos", targets)
        completed = len(self.robot.writes)
        for _ in range(30):
            self.tick(None, face=True)
        self.assertGreater(len(self.robot.writes), completed)
        self.assertEqual(self.engine._companion_state, "moving")

    def test_rejected_agent_plan_keeps_companion_enabled_and_other_axes_running(self):
        self.prepare_near_elbow_boundary()
        for _ in range(20):
            self.tick(None, face=True)
        self.robot.positions["elbow_flex.pos"] = -70.1
        with self.assertRaisesRegex(RuntimeError, "elbow_flex"):
            self.engine._start_agent_plan(
                [{"name": "nod", "arguments": {}}],
                client_id="operator", submitted_epoch=self.engine._epoch,
            )
        before = len(self.robot.writes)
        for _ in range(30):
            self.tick(None, face=True)
        self.assertEqual(self.engine._companion_client, "operator")
        self.assertGreater(len(self.robot.writes), before)
        self.assertIsNone(self.engine._agent_plan)

    def test_stop_still_disables_companion_and_blocks_future_ticks(self):
        for _ in range(20):
            self.tick(None, face=True)
        self.engine.disarm("operator pressed stop-following")
        before = len(self.robot.writes)
        for _ in range(20):
            self.tick(None, face=True)
        self.assertIsNone(self.engine._companion_client)
        self.assertEqual(len(self.robot.writes), before)

    def test_large_unmeasured_geometry_change_holds_without_sending_empty_goals(self):
        for joint in ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"):
            self.robot.positions[joint] = self.engine.agent_joint_limits()[joint][1] + 0.1
        for _ in range(20):
            self.tick(None, face=True)
        self.assertEqual(self.robot.writes, [])
        self.assertEqual(self.engine._companion_state, "gaze_required")
        self.assertEqual(self.engine._gaze_state, "pose_changed")
        self.assertEqual(self.engine._companion_client, "operator")
        self.assertIsNone(self.engine._error)

        reads = self.robot.read_count
        for _ in range(10):
            self.clock += 0.005
            self.engine._companion_frame_at = self.clock
            self.engine._poll_companion(self.clock)
        self.assertLessEqual(self.robot.read_count - reads, 2)

    def test_message_planning_and_deferred_reply_keep_companion_moving(self):
        self.client_id = "companion_agent_test_operator"
        self.engine._operator_client = self.client_id
        self.engine._camera_owner = self.client_id
        self.engine._companion_client = self.client_id
        self.engine.heartbeat(self.client_id)
        self.tick(None, face=True)

        def submit_text(text, *, context):
            before = len(self.robot.writes)
            # Simulate time spent awaiting a model or a text-only response.
            for _ in range(20):
                self.tick(None, face=True)
            self.assertGreater(len(self.robot.writes), before)
            self.assertEqual(self.engine._companion_client, self.client_id)
            return {"task_id": "test-task"}

        for state in ("READY", "COMPLETE"):
            with self.subTest(task_state=state), tempfile.TemporaryDirectory() as directory:
                service = SimpleNamespace(
                    submit_text=submit_text,
                    get_task=lambda task_id: {"task_id": task_id, "state": state, "plan": {"actions": []}},
                )
                app = create_app(self.engine, agent_service=service,
                                 preferences_path=Path(directory) / "preferences.json")
                app.config.update(TESTING=True)
                client = app.test_client()
                client.get("/")
                reply = client.post(
                    "/api/agent/message", json={"text": "How are you?", "defer_motion": True},
                    headers={"X-Client-Id": self.client_id, "Origin": "http://localhost"},
                )
                self.assertEqual(reply.status_code, 200)
                self.assertTrue(reply.get_json()["ok"])
                self.assertIsNone(self.engine._agent_plan)

    def test_agent_interruption_keeps_companion_enabled_without_long_cooldown(self):
        self.client_id = "companion_agent_test_operator"
        self.engine._operator_client = self.client_id
        self.engine._camera_owner = self.client_id
        self.engine._companion_client = self.client_id
        self.engine.heartbeat(self.client_id)
        for _ in range(20):
            self.tick(None, face=True)
        service = SimpleNamespace(
            interruption_resume=lambda client_id: None,
            interruption_context=lambda client_id, **kwargs: None,
            cancel_all=lambda reason: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(self.engine, agent_service=service,
                             preferences_path=Path(directory) / "preferences.json")
            app.config.update(TESTING=True)
            client = app.test_client()
            client.get("/")
            reply = client.post(
                "/api/agent/interrupt", json={},
                headers={"X-Client-Id": self.client_id, "Origin": "http://localhost"},
            )
            self.assertEqual(reply.status_code, 200)
            self.assertEqual(self.engine._companion_client, self.client_id)
            before = len(self.robot.writes)
            for _ in range(20):
                self.tick(None, face=True)
            self.assertGreater(len(self.robot.writes), before)
            self.assertEqual(self.engine._companion_client, self.client_id)
