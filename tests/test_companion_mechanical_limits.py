import unittest
from types import SimpleNamespace

from so101_human_follow.companion import CompanionMotion
from so101_human_follow.config import ALL_JOINTS, load_config
from so101_human_follow.robot_adapter import SO101RobotAdapter, _make_lerobot_importable, preview_initial_positions


_make_lerobot_importable()
from lerobot.motors import Motor, MotorCalibration, MotorNormMode


class NoIOBus:
    is_connected = True

    def __init__(self):
        self.motors = {
            joint.removesuffix(".pos"): Motor(
                index, "sts3215", MotorNormMode.RANGE_0_100 if joint == "gripper.pos" else MotorNormMode.RANGE_M100_100,
            ) for index, joint in enumerate(ALL_JOINTS, 1)
        }
        self.calibration = {
            name: MotorCalibration(motor.id, 0, 1000, 300, 3700) for name, motor in self.motors.items()
        }

    @property
    def is_calibrated(self):
        raise AssertionError("normalized_joint_limits must not read EEPROM")

    def read_calibration(self):
        raise AssertionError("normalized_joint_limits must not read EEPROM")

    def sync_read(self, *args, **kwargs):
        raise AssertionError("normalized_joint_limits must not read positions")

    def sync_write(self, *args, **kwargs):
        raise AssertionError("normalized_joint_limits must not write motors")


class CompanionMechanicalLimitsTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.example.json")
        self.adapter = SO101RobotAdapter(self.config.robot)
        self.bus = NoIOBus()
        self.adapter.robot = SimpleNamespace(bus=self.bus, calibration=dict(self.bus.calibration))
        self.adapter.connected = True
        self.adapter._verified_calibration_fingerprint = self.adapter.calibration_fingerprint()

    def test_reads_actual_bus_normalization_for_all_six_axes_without_io(self):
        result = self.adapter.normalized_joint_limits()
        self.assertEqual(set(result), set(ALL_JOINTS))
        self.assertEqual(result["gripper.pos"], (0.0, 100.0))
        for joint in ALL_JOINTS[:-1]:
            self.assertEqual(result[joint], (-100.0, 100.0))

    def test_drive_direction_does_not_expand_or_shift_bounds(self):
        for calibration in self.bus.calibration.values():
            calibration.drive_mode = 1
        self.adapter._verified_calibration_fingerprint = self.adapter.calibration_fingerprint()
        result = self.adapter.normalized_joint_limits()
        self.assertEqual(result["shoulder_pan.pos"], (-100.0, 100.0))
        self.assertEqual(result["gripper.pos"], (0.0, 100.0))

    def test_disconnected_unverified_or_missing_calibration_fails_closed(self):
        self.adapter.connected = False
        with self.assertRaises(RuntimeError):
            self.adapter.normalized_joint_limits()
        self.adapter.connected = True
        self.adapter._verified_calibration_fingerprint = None
        with self.assertRaises(RuntimeError):
            self.adapter.normalized_joint_limits()
        self.adapter._verified_calibration_fingerprint = self.adapter.calibration_fingerprint()
        self.bus.is_connected = False
        with self.assertRaises(RuntimeError):
            self.adapter.normalized_joint_limits()
        self.bus.is_connected = True
        del self.bus.calibration["elbow_flex"]
        with self.assertRaises(RuntimeError):
            self.adapter.normalized_joint_limits()

    def test_changed_calibration_and_invalid_raw_range_fail_closed(self):
        self.bus.calibration["wrist_flex"].range_max = 3600
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.adapter.normalized_joint_limits()
        self.bus.calibration["wrist_flex"].range_max = 300
        self.adapter._verified_calibration_fingerprint = self.adapter.calibration_fingerprint()
        with self.assertRaisesRegex(RuntimeError, "Invalid calibrated"):
            self.adapter.normalized_joint_limits()

    def test_degrees_radians_strings_and_unknown_modes_are_not_guessed(self):
        for unsupported in (MotorNormMode.DEGREES, "radians", "range_m100_100", None):
            with self.subTest(mode=unsupported):
                self.bus.motors["wrist_flex"].norm_mode = unsupported
                with self.assertRaisesRegex(RuntimeError, "Unsupported motor normalization"):
                    self.adapter.normalized_joint_limits()

    def test_relative_companion_starts_from_comfortable_pose_outside_mimic_limit(self):
        original_limits = dict(self.config.safety.absolute_limits)
        current = preview_initial_positions()
        current["shoulder_pan.pos"] = -81.77
        mechanical = self.adapter.normalized_joint_limits()
        motion = CompanionMotion(current, mechanical)
        self.assertTrue(motion.tracks_yaw)
        self.assertNotIn("shoulder_pan.pos", motion.blocked_joints)
        low, high = motion.gaze_bounds["shoulder_pan.pos"]
        self.assertAlmostEqual(high, -61.77)
        self.assertGreater(low, -100.0)
        previous = current["shoulder_pan.pos"]
        for _ in range(90):
            target = motion.step(0.05, face_target=(0.7, 0.5), gaze_delta=(-0.5, 0), max_step=0.5)
            yaw = target["shoulder_pan.pos"]
            self.assertGreaterEqual(yaw, low)
            self.assertLessEqual(yaw, high)
            self.assertLessEqual(abs(yaw - previous), 0.500001)
            previous = yaw
        self.assertEqual(self.config.safety.absolute_limits, original_limits)
        self.assertEqual(self.config.safety.absolute_limits["shoulder_pan.pos"], (-70.0, 70.0))

    def test_pose_outside_calibrated_motor_range_remains_blocked(self):
        current = preview_initial_positions()
        current["shoulder_pan.pos"] = -100.5
        motion = CompanionMotion(current, self.adapter.normalized_joint_limits())
        self.assertFalse(motion.tracks_yaw)
        self.assertIn("shoulder_pan.pos", motion.blocked_joints)

    def test_disconnect_clears_calibration_verification(self):
        self.adapter.robot = None
        self.adapter._cleanup_partial_connection()
        self.assertIsNone(self.adapter._verified_calibration_fingerprint)
        with self.assertRaises(RuntimeError):
            self.adapter.normalized_joint_limits()


if __name__ == "__main__":
    unittest.main()
