import time
import unittest
from types import SimpleNamespace

from so101_human_follow.agent_actions import VisualProximity
from so101_human_follow.companion import CompanionMotion, face_center
from so101_human_follow.config import ALL_JOINTS, load_config
from so101_human_follow.gaze_controller import GazeCalibration, ImageGazeController
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.pose_math import Landmark
from so101_human_follow.web_engine import ControlEngine


YAW = "shoulder_pan.pos"


class CompanionTrackingTest(unittest.TestCase):
    def test_face_target_ignores_shoulders_and_offscreen_landmarks(self):
        points = [Landmark(0.1, 0.8, visibility=0.1) for _ in range(33)]
        points[11] = Landmark(0.2, 0.8)
        points[12] = Landmark(0.6, 0.8)
        self.assertIsNone(face_center(points))
        points[0] = Landmark(0.7, 0.2)
        points[2] = Landmark(0.6, 0.2)
        points[5] = Landmark(1.2, 0.2)
        x, y = face_center(points)
        self.assertAlmostEqual(x, 0.65)
        self.assertAlmostEqual(y, 0.2)

    def test_search_sweeps_only_servo_one_and_reverses_within_limits(self):
        limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        positions = dict.fromkeys(ALL_JOINTS, 0.0)
        positions[YAW] = 58.0
        motion = CompanionMotion(positions, limits)
        previous = positions[YAW]
        samples = []
        for _ in range(100):
            targets = motion.step(0.5, face_target=None, search=True, max_step=0.8)
            self.assertEqual(set(targets), {YAW})
            position = targets[YAW]
            self.assertLessEqual(abs(position - previous), 0.800001)
            self.assertTrue(38.0 <= position <= 70.0)
            samples.append(position)
            previous = position
        self.assertIn(38.0, samples)
        self.assertLess(max(samples), 70.0)
        self.assertTrue(any(after > before for before, after in zip(samples, samples[1:])))
        self.assertTrue(any(after < before for before, after in zip(samples, samples[1:])))
        targets = motion.step(0.05, face_target=(0.5, 0.5), max_step=0.8)
        self.assertAlmostEqual(targets[YAW], previous)

    def test_out_of_range_body_axes_are_omitted_without_changing_feedback(self):
        limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        limits["gripper.pos"] = (0.0, 100.0)
        for joint in ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"):
            for measured in (-70.09936766034326, 70.189701897):
                with self.subTest(joint=joint, measured=measured):
                    positions = dict.fromkeys(ALL_JOINTS, 0.0)
                    positions["gripper.pos"] = 50.0
                    positions[joint] = measured
                    motion = CompanionMotion(positions, limits)
                    moved = False
                    for _ in range(80):
                        targets = motion.step(0.05, face_target=(0.6, 0.4), max_step=0.8)
                        self.assertNotIn(joint, targets)
                        moved |= any(abs(value) > 0.1 for value in targets.values())
                        for active, value in targets.items():
                            self.assertTrue(limits[active][0] <= value <= limits[active][1])
                    self.assertTrue(moved)
                    self.assertEqual(positions[joint], measured)
                    self.assertIn(joint, motion.blocked_joints)

    def test_continuous_body_goals_leave_room_before_calibrated_endpoints(self):
        positions = dict.fromkeys(ALL_JOINTS, 0.0)
        limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        positions["elbow_flex.pos"] = -69.9
        motion = CompanionMotion(positions, limits)
        previous = positions["elbow_flex.pos"]
        samples = []
        for _ in range(300):
            targets = motion.step(0.05, face_target=(0.5, 0.5), max_step=0.8)
            value = targets["elbow_flex.pos"]
            self.assertLessEqual(abs(value - previous), 0.800001)
            self.assertGreater(value, -70.0)
            previous = value
            samples.append(value)
        self.assertGreater(min(samples[-150:]), -69.0)

    def test_missing_encoder_feedback_is_still_rejected(self):
        positions = dict.fromkeys(ALL_JOINTS, 0.0)
        limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        positions["elbow_flex.pos"] = float("nan")
        with self.assertRaisesRegex(ValueError, "有效的实测位置"):
            CompanionMotion(positions, limits)

    def test_control_tick_search_is_opt_in_and_stops_on_stale_camera(self):
        config = load_config("config.example.json")
        engine = ControlEngine(config, robot_allowed=True)
        client = "client_companion_tracking_test"
        engine.heartbeat(client)
        now = time.monotonic()
        positions = dict.fromkeys(ALL_JOINTS, 0.0)
        sent = []

        def observation():
            normalized = dict(positions)
            raw = {joint: value * 20.0 if joint == "gripper.pos" else (value + 100.0) * 10.0
                   for joint, value in normalized.items()}
            limits = {joint: (0.0, 2000.0) for joint in normalized}
            return {"positions": normalized, "raw_positions": raw, "raw_limits": limits,
                    "clipped_joints": tuple(joint for joint, value in raw.items()
                                            if not limits[joint][0] <= value <= limits[joint][1])}

        def send(targets, active, fixed, *, before_write, verify_calibrated_raw=False):
            self.assertTrue(verify_calibrated_raw)
            self.assertFalse(observation()["clipped_joints"])
            for joint, value in {**fixed, **{joint: targets[joint] for joint in active}}.items():
                low, high = (0.0, 100.0) if joint == "gripper.pos" else (-100.0, 100.0)
                self.assertTrue(low <= value <= high, (joint, value))
            if before_write(positions):
                sent.append(dict(targets))
            return positions, positions

        engine._robot = SimpleNamespace(
            connected=True, read_positions=lambda: positions, send_agent_targets=send,
            read_calibration_observation=observation, hold_current=lambda: dict(positions),
            hold_raw_current=lambda: dict(positions),
        )
        engine._robot_state = "CONNECTED_DISARMED"
        engine._safe_profile = SimpleNamespace(controlled_limits={})
        engine._companion_mechanical_limits = engine.agent_joint_limits()
        engine._camera_owner = client
        engine._camera_name = config.camera.wrist_name
        engine._preview_state = "RUNNING"
        engine._companion_frame_at = now
        engine._gaze_calibration = GazeCalibration(
            [[-0.02, 0.0], [0.0, -0.02]], positions, engine._gaze_metadata_locked(),
        )
        engine._gaze_controller = ImageGazeController(engine._gaze_calibration)
        engine._gaze_state = "ready"
        engine._companion_client = client
        engine._detection = PoseDetection(None, (), 0, "no person")
        engine._visual_proximity = VisualProximity("UNKNOWN", "no reliable visual scale")

        engine._poll_companion(now)
        self.assertEqual(engine._companion_state, "waiting_person")
        self.assertFalse(sent)
        engine._companion_allow_search = True
        engine._poll_companion(now)
        self.assertEqual(engine._companion_state, "checking_presence")
        self.assertFalse(sent)
        engine._poll_companion(now + 0.46)
        self.assertEqual(engine._companion_state, "searching")
        self.assertEqual([set(targets) for targets in sent], [{YAW}])
        engine._poll_companion(now + 1.0)
        self.assertEqual(engine._companion_state, "waiting_camera")
        self.assertTrue(engine._agent_hold_pending)
        self.assertEqual(len(sent), 1)

        engine._hold_agent_if_pending()
        engine._companion_motion = None
        engine._companion_frame_at = now + 1.1
        positions[YAW] = -78.2
        engine._poll_companion(now + 1.1)
        engine._poll_companion(now + 1.56)
        self.assertEqual(engine._companion_state, "search_blocked")
        self.assertIn("-78.2", engine._companion_reason)
        self.assertFalse(engine._agent_hold_pending)
        self.assertEqual(len(sent), 1)
