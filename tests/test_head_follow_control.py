import unittest
from types import SimpleNamespace
from unittest.mock import patch

from so101_human_follow.agent_actions import estimate_visual_proximity
from so101_human_follow.config import load_config
from so101_human_follow.gaze_controller import GazeCalibration, ImageGazeController
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.robot_adapter import preview_initial_positions
from so101_human_follow.web_engine import ControlEngine


def head_detection(center=(0.25, 0.75), box=(0.18, 0.66, 0.32, 0.84), **changes):
    fields = dict(
        features=None, landmarks=(), people_count=0, reason="no person",
        head_center=center, head_box=box, head_confidence=0.9, head_source="face",
    )
    fields.update(changes)
    return PoseDetection(**fields)


class HeadProximityTest(unittest.TestCase):
    def test_head_only_provides_apparent_size_without_body_control_channels(self):
        detection = head_detection()
        proximity = estimate_visual_proximity(detection, aspect_ratio=4 / 3)
        self.assertEqual(proximity.state, "SAFE")
        self.assertIsNotNone(proximity.head_size)
        self.assertIsNone(proximity.shoulder_size)
        self.assertIsNone(proximity.palm_size)
        self.assertIsNone(detection.features)
        self.assertEqual(detection.people_count, 0)

    def test_large_head_still_triggers_the_close_stop(self):
        proximity = estimate_visual_proximity(
            head_detection((0.5, 0.5), (0.1, 0.1, 0.9, 0.9)), aspect_ratio=4 / 3,
        )
        self.assertEqual(proximity.state, "TOO_CLOSE")

    def test_invalid_or_low_confidence_head_does_not_invent_a_safe_distance(self):
        for changes in (
            {"head_confidence": 0.2}, {"head_confidence": 1.1}, {"head_confidence": float("nan")},
            {"head_box": (0.3, 0.1, 0.2, 0.3)},
            {"head_box": (0.0, 0.0, float("nan"), 0.8)},
            {"head_center": (0.9, 0.9)}, {"head_box": None},
        ):
            with self.subTest(changes=changes):
                proximity = estimate_visual_proximity(head_detection(**changes), aspect_ratio=4 / 3)
                self.assertEqual(proximity.state, "UNKNOWN")


class HeadFollowControlTest(unittest.TestCase):
    def setUp(self):
        self.clock = 10.0
        timer = patch("so101_human_follow.web_engine.time.monotonic", side_effect=lambda: self.clock)
        timer.start()
        self.addCleanup(timer.stop)
        self.config = load_config("config.example.json")
        self.engine = ControlEngine(self.config, robot_allowed=True)
        self.client = "head_tracking_test_operator"
        self.positions = preview_initial_positions()
        self.writes = []
        self.holds = 0

        def hold():
            self.holds += 1
            return dict(self.positions)

        def observation():
            positions = dict(self.positions)
            raw = {joint: value * 20.0 if joint == "gripper.pos" else (value + 100.0) * 10.0
                   for joint, value in positions.items()}
            limits = {joint: (0.0, 2000.0) for joint in positions}
            return {"positions": positions, "raw_positions": raw, "raw_limits": limits,
                    "clipped_joints": tuple(joint for joint, value in raw.items()
                                            if not limits[joint][0] <= value <= limits[joint][1])}

        def send(targets, active, fixed, *, before_write, verify_calibrated_raw=False):
            self.assertTrue(verify_calibrated_raw)
            self.assertFalse(observation()["clipped_joints"])
            for joint, value in {**fixed, **{joint: targets[joint] for joint in active}}.items():
                low, high = (0.0, 100.0) if joint == "gripper.pos" else (-100.0, 100.0)
                self.assertTrue(low <= value <= high, (joint, value))
            observed = dict(self.positions)
            if before_write(observed):
                self.writes.append({joint: targets[joint] for joint in active})
                self.positions.update(self.writes[-1])
            return dict(self.positions), observed

        self.engine._robot = SimpleNamespace(
            connected=True, read_positions=lambda: dict(self.positions), hold_current=hold,
            send_agent_targets=send, read_calibration_observation=observation, hold_raw_current=hold,
        )
        self.engine._robot_state = "CONNECTED_DISARMED"
        self.engine._safe_profile = SimpleNamespace(
            controlled_limits=dict(self.config.safety.absolute_limits), fixed_positions={},
        )
        self.engine._companion_mechanical_limits = self.engine.agent_joint_limits()
        self.engine.heartbeat(self.client)
        self.engine._camera_owner = self.client
        self.engine._camera_name = self.config.camera.wrist_name
        self.engine._preview_state = "RUNNING"
        self.engine._companion_frame_at = self.clock
        self.engine._gaze_calibration = GazeCalibration(
            [[-0.02, 0.0], [0.0, -0.02]], self.positions, self.engine._gaze_metadata_locked(),
        )
        self.engine._gaze_controller = ImageGazeController(self.engine._gaze_calibration)
        self.engine._gaze_state = "ready"
        self.engine.set_companion(
            client_id=self.client, enabled=True, allow_unoccupied_idle=False,
            expected_epoch=self.engine._epoch,
        )

    def tick(self, detection, dt=0.06):
        self.clock += dt
        self.engine.heartbeat(self.client)
        self.engine._update_tracking(detection, self.clock, captured_at=self.clock)
        self.engine._hold_agent_if_pending()
        self.engine._poll_companion(self.clock)

    def acquire_head(self):
        for _ in range(15):
            self.tick(head_detection())

    def test_camera_centers_head_even_without_shoulders_hands_or_arm_features(self):
        self.acquire_head()
        self.assertEqual(self.engine._companion_state, "moving")
        self.assertGreater(len(self.writes), 0)
        self.assertLess(self.positions["shoulder_pan.pos"], 0.0)
        self.assertGreater(self.positions["wrist_flex.pos"], 0.0)
        status = self.engine.status(self.client)
        self.assertTrue(status["companion"]["head_visible"])
        self.assertEqual(status["companion"]["head_center"], (0.25, 0.75))
        self.assertEqual(status["valid_channels"], [])

    def test_brief_head_loss_holds_then_resumes_without_reconfirming_every_frame(self):
        self.acquire_head()
        before = len(self.writes)
        self.tick(PoseDetection(None, (), 0, "no person"))
        self.assertEqual(len(self.writes), before)
        self.assertTrue(self.engine._agent_hold_pending)
        self.tick(head_detection())
        self.assertEqual(len(self.writes), before)
        self.tick(head_detection())
        self.assertGreater(len(self.writes), before)
        self.assertEqual(self.engine._companion_state, "moving")
        self.assertGreater(self.holds, 0)

    def test_long_head_loss_requires_confirmation_again(self):
        self.acquire_head()
        for _ in range(5):
            self.tick(PoseDetection(None, (), 0, "no person"))
        before = len(self.writes)
        self.tick(head_detection())
        self.assertEqual(len(self.writes), before)
        self.assertEqual(self.engine._companion_state, "checking_presence")

    def test_close_or_ambiguous_target_stops_instead_of_looking_for_another_head(self):
        for detection, expected in (
            (head_detection((0.5, 0.5), (0.1, 0.1, 0.9, 0.9)), "too_close"),
            (head_detection(head_center=None, head_box=None, head_source="ambiguous"), "ambiguous"),
            (head_detection(people_count=2), "ambiguous"),
        ):
            with self.subTest(expected=expected):
                self.acquire_head()
                before = len(self.writes)
                self.tick(detection)
                self.assertEqual(len(self.writes), before)
                self.assertEqual(self.engine._companion_state, expected)
                self.assertIsNone(self.engine._companion_presence)
                self.assertEqual(self.engine._companion_client, self.client)

    def test_stale_camera_still_stops_head_following(self):
        self.acquire_head()
        before = len(self.writes)
        remaining = self.config.safety.tracking_timeout_s + 0.1
        while remaining > 0.0:
            step = min(0.1, remaining)
            self.clock += step
            self.engine.heartbeat(self.client)
            remaining -= step
        self.engine._poll_companion(self.clock)
        self.assertEqual(len(self.writes), before)
        self.assertEqual(self.engine._companion_state, "waiting_camera")
        self.assertTrue(self.engine._agent_hold_pending)

    def test_legacy_search_request_keeps_vertical_search_off(self):
        self.engine.set_companion(
            client_id=self.client, enabled=True, allow_unoccupied_idle=True,
            expected_epoch=self.engine._epoch,
        )
        for _ in range(20):
            self.tick(PoseDetection(None, (), 0, "no person"))
        self.assertTrue(self.writes)
        self.assertFalse(self.engine._companion_allow_vertical_search)
        for write in self.writes:
            self.assertEqual(set(write), {"shoulder_pan.pos"})
        before = len(self.writes)
        self.tick(PoseDetection(None, (), 1, "no reliable control channel"))
        self.assertEqual(len(self.writes), before)
        self.assertEqual(self.engine._companion_state, "waiting_face")

    def test_explicit_search_can_bring_an_offscreen_head_into_view(self):
        self.engine.set_companion(
            client_id=self.client, enabled=True, allow_unoccupied_idle=True,
            allow_vertical_search=True, expected_epoch=self.engine._epoch,
        )
        # A body candidate with no reliable face must not permanently prevent
        # the explicitly requested search for the operator's head.
        for _ in range(20):
            self.tick(PoseDetection(None, (), 1, "no reliable control channel"))
        self.assertEqual(self.engine._companion_state, "searching")
        self.assertTrue(self.writes)
        for write in self.writes:
            self.assertEqual(set(write), {"shoulder_pan.pos", "wrist_flex.pos"})
            for joint, target in write.items():
                low, high = self.engine.agent_joint_limits()[joint]
                self.assertTrue(low <= target <= high)
        before = len(self.writes)
        self.tick(head_detection())
        self.assertEqual(self.engine._companion_state, "checking_presence")
        self.assertEqual(len(self.writes), before)
        for _ in range(15):
            self.tick(head_detection())
        self.assertEqual(self.engine._companion_state, "moving")
        self.assertEqual(self.engine._companion_client, self.client)


if __name__ == "__main__":
    unittest.main()
