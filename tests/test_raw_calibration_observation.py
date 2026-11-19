from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from so101_human_follow.config import ALL_JOINTS, load_config
from so101_human_follow.robot_adapter import (
    CalibrationRangeError,
    SO101RobotAdapter,
    _make_lerobot_importable,
    preview_initial_positions,
)


_make_lerobot_importable()
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import MotorsBus


WRIST = "wrist_flex.pos"
YAW = "shoulder_pan.pos"


class NoDeviceFeetechBus:
    """Use the SDK's register/normalization code with only the I/O replaced."""

    is_connected = True
    apply_drive_mode = True
    protocol_version = 0
    _has_different_ctrl_tables = False
    model_ctrl_table = FeetechMotorsBus.model_ctrl_table
    model_encoding_table = FeetechMotorsBus.model_encoding_table
    normalized_data = FeetechMotorsBus.normalized_data
    _get_motors_list = MotorsBus._get_motors_list
    _get_ids_values_dict = MotorsBus._get_ids_values_dict
    _assert_protocol_is_compatible = FeetechMotorsBus._assert_protocol_is_compatible
    _decode_sign = FeetechMotorsBus._decode_sign
    _encode_sign = FeetechMotorsBus._encode_sign

    def __init__(self):
        self.motors = {
            joint.removesuffix(".pos"): Motor(
                index, "sts3215",
                MotorNormMode.RANGE_0_100 if joint == "gripper.pos" else MotorNormMode.RANGE_M100_100,
            ) for index, joint in enumerate(ALL_JOINTS, 1)
        }
        self.calibration = {
            name: MotorCalibration(motor.id, 0, 173, 1000, 3000)
            for name, motor in self.motors.items()
        }
        self.raw = dict.fromkeys(self.motors, 2000)
        self.read_calls = []
        self.normalized_batches = []
        self.write_calls = []
        self.wire_writes = []
        self.read_error = None
        self.after_read = None

    @property
    def ids(self):
        return [motor.id for motor in self.motors.values()]

    def _id_to_name(self, motor_id):
        return next(name for name, motor in self.motors.items() if motor.id == motor_id)

    def _id_to_model(self, motor_id):
        return self.motors[self._id_to_name(motor_id)].model

    def sync_read(self, data_name, motors=None, **kwargs):
        self.read_calls.append((data_name, motors, dict(kwargs)))
        return MotorsBus.sync_read(self, data_name, motors, **kwargs)

    def _sync_read(self, address, length, motor_ids, **kwargs):
        if self.read_error is not None:
            raise self.read_error
        values = {
            motor_id: self.raw[self._id_to_name(motor_id)]
            for motor_id in motor_ids if self._id_to_name(motor_id) in self.raw
        }
        if self.after_read is not None:
            self.after_read()
        return values, 0

    def _normalize(self, values):
        self.normalized_batches.append(dict(values))
        return MotorsBus._normalize(self, values)

    def _unnormalize(self, values):
        return MotorsBus._unnormalize(self, values)

    def sync_write(self, data_name, values, **kwargs):
        self.write_calls.append((data_name, dict(values), dict(kwargs)))
        return MotorsBus.sync_write(self, data_name, values, **kwargs)

    def _sync_write(self, address, length, values, **kwargs):
        self.wire_writes.append(dict(values))
        return 0

    @property
    def is_calibrated(self):
        raise AssertionError("The raw observation must not read EEPROM calibration.")

    def read_calibration(self):
        raise AssertionError("The raw observation must not read EEPROM calibration.")

    def read(self, *args, **kwargs):
        raise AssertionError("Only one raw sync_read is allowed.")

    def write(self, *args, **kwargs):
        raise AssertionError("Calibration and torque registers must not be written.")

    def enable_torque(self, *args, **kwargs):
        raise AssertionError("These operations must not enable torque.")

    def disable_torque(self, *args, **kwargs):
        raise AssertionError("These operations must not disable torque.")


class RawCalibrationObservationTest(unittest.TestCase):
    def setUp(self):
        self.adapter = SO101RobotAdapter(load_config("config.example.json").robot)
        self.bus = NoDeviceFeetechBus()
        self.robot = SimpleNamespace(
            bus=self.bus, calibration=deepcopy(self.bus.calibration),
            get_observation=Mock(side_effect=AssertionError("No second normalized observation is allowed.")),
        )
        self.adapter.robot = self.robot
        self.adapter.connected = True
        self.mark_verified()

    def mark_verified(self):
        self.adapter._verified_calibration_fingerprint = self.adapter.calibration_fingerprint()

    def assert_one_raw_batch(self):
        self.assertEqual(len(self.bus.read_calls), 1)
        register, motors, kwargs = self.bus.read_calls[0]
        self.assertEqual(register, "Present_Position")
        self.assertEqual(motors, [joint.removesuffix(".pos") for joint in ALL_JOINTS])
        self.assertEqual(kwargs, {"normalize": False, "num_retry": 0})
        self.robot.get_observation.assert_not_called()

    def test_raw_evidence_distinguishes_clipped_values_from_legitimate_endpoints(self):
        for raw, normalized, in_range in (
            (999, -100.0, False), (1000, -100.0, True), (2000, 0.0, True),
            (3000, 100.0, True), (3001, 100.0, False),
        ):
            with self.subTest(raw=raw):
                self.bus.read_calls.clear()
                self.bus.normalized_batches.clear()
                self.bus.raw["wrist_flex"] = raw
                observation = self.adapter.read_calibration_observation()
                self.assert_one_raw_batch()
                self.assertEqual(set(observation["positions"]), set(ALL_JOINTS))
                self.assertTrue(all(type(value) is float for value in observation["positions"].values()))
                self.assertEqual(observation["positions"][WRIST], normalized)
                self.assertEqual(observation["raw_positions"][WRIST], raw)
                self.assertEqual(observation["raw_limits"][WRIST], (1000, 3000))
                self.assertEqual(observation["in_range"][WRIST], in_range)
                self.assertEqual(observation["clipped_joints"], () if in_range else (WRIST,))
                self.assertEqual(self.bus.normalized_batches[0][4], raw)
                self.assertEqual(self.bus.write_calls, [])

    def test_normalization_uses_sdk_drive_direction_without_reapplying_homing(self):
        for name in ("wrist_flex", "gripper"):
            self.bus.calibration[name].drive_mode = 1
            self.robot.calibration[name].drive_mode = 1
            self.bus.raw[name] = 1000
        self.mark_verified()
        observation = self.adapter.read_calibration_observation()
        self.assert_one_raw_batch()
        self.assertEqual(observation["raw_positions"][WRIST], 1000)
        self.assertEqual(observation["positions"][WRIST], 100.0)
        self.assertEqual(observation["positions"]["gripper.pos"], 100.0)
        self.assertEqual(observation["clipped_joints"], ())
        self.assertEqual(self.bus.calibration["wrist_flex"].homing_offset, 173)
        self.assertEqual(self.bus.write_calls, [])

    def test_normalized_and_raw_positions_come_from_the_same_time_sample(self):
        self.bus.after_read = lambda: self.bus.raw.update(wrist_flex=3000)
        observation = self.adapter.read_calibration_observation()
        self.assert_one_raw_batch()
        self.assertEqual(observation["raw_positions"][WRIST], 2000)
        self.assertEqual(observation["positions"][WRIST], 0.0)
        self.assertEqual(self.bus.raw["wrist_flex"], 3000)

    def test_unconnected_or_unverified_observation_fails_before_io(self):
        self.adapter.connected = False
        with self.assertRaises(RuntimeError):
            self.adapter.read_calibration_observation()
        self.adapter.connected = True
        self.adapter._verified_calibration_fingerprint = None
        with self.assertRaises(RuntimeError):
            self.adapter.read_calibration_observation()
        self.mark_verified()
        self.bus.is_connected = False
        with self.assertRaises(RuntimeError):
            self.adapter.read_calibration_observation()
        self.assertEqual(self.bus.read_calls, [])
        self.assertEqual(self.bus.write_calls, [])

    def test_unknown_normalization_or_changed_calibration_fails_before_io(self):
        original = self.bus.motors["wrist_flex"].norm_mode
        for mode in (MotorNormMode.DEGREES, "range_m100_100", None):
            with self.subTest(mode=mode):
                self.bus.motors["wrist_flex"].norm_mode = mode
                with self.assertRaisesRegex(RuntimeError, "Unsupported motor normalization"):
                    self.adapter.read_calibration_observation()
        self.bus.motors["wrist_flex"].norm_mode = original
        self.robot.calibration["wrist_flex"].homing_offset += 1
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.adapter.read_calibration_observation()
        self.assertEqual(self.bus.read_calls, [])

    def test_unknown_drive_mode_behavior_is_not_guessed(self):
        self.bus.apply_drive_mode = False
        with self.assertRaisesRegex(RuntimeError, "Feetech drive-mode"):
            self.adapter.read_calibration_observation()
        self.assertEqual(self.bus.read_calls, [])

    def test_calibration_changes_during_read_are_rejected_without_writes(self):
        self.bus.after_read = lambda: setattr(self.robot.calibration["wrist_flex"], "range_min", 1001)
        with self.assertRaisesRegex(RuntimeError, "calibration changed"):
            self.adapter.read_calibration_observation()
        self.assert_one_raw_batch()
        self.assertEqual(self.bus.write_calls, [])

    def test_read_failure_does_not_fall_back_to_another_sample(self):
        self.bus.read_error = ConnectionError("simulated raw read failure")
        with self.assertRaisesRegex(RuntimeError, "simulated raw read failure"):
            self.adapter.read_calibration_observation()
        self.assert_one_raw_batch()
        self.assertEqual(self.bus.normalized_batches, [])
        self.assertEqual(self.bus.write_calls, [])

    def test_incomplete_and_noninteger_raw_samples_fail_closed(self):
        original = dict(self.bus.raw)
        for invalid in (None, True, 1000.25, float("nan"), "1000"):
            with self.subTest(raw=invalid):
                self.bus.raw = {**original, "wrist_flex": invalid}
                with self.assertRaisesRegex(RuntimeError, "integer encoder"):
                    self.adapter.read_calibration_observation()
        self.bus.raw = {name: value for name, value in original.items() if name != "wrist_flex"}
        with self.assertRaisesRegex(RuntimeError, "complete six-joint"):
            self.adapter.read_calibration_observation()
        self.assertEqual(self.bus.write_calls, [])

    def test_invalid_normalization_results_fail_closed(self):
        for result in ({}, {motor.id: float("nan") for motor in self.bus.motors.values()}):
            with self.subTest(result=result), patch.object(self.bus, "_normalize", return_value=result):
                with self.assertRaises(RuntimeError):
                    self.adapter.read_calibration_observation()
        self.assertEqual(self.bus.write_calls, [])

    def test_raw_gated_send_rejects_active_and_fixed_clipped_axes_before_any_write(self):
        self.bus.raw["wrist_flex"] = 999
        callback = Mock(return_value=True)
        for active, fixed in (((WRIST,), {}), ((YAW,), {WRIST: -100.0})):
            with self.subTest(active=active, fixed=fixed):
                self.bus.read_calls.clear()
                with self.assertRaises(CalibrationRangeError) as caught:
                    self.adapter.send_agent_targets(
                        {WRIST: -97.0, YAW: 2.0}, active, fixed,
                        before_write=callback, verify_calibrated_raw=True,
                    )
                self.assert_one_raw_batch()
                self.assertEqual(caught.exception.joints, (WRIST,))
                self.assertEqual(caught.exception.observation["raw_positions"][WRIST], 999)
                self.assertFalse(caught.exception.observation["in_range"][WRIST])
                self.assertIn("raw 999", str(caught.exception))
                callback.assert_not_called()
                self.assertEqual(self.bus.write_calls, [])

    def test_raw_gated_endpoint_probe_moves_inward_using_one_shared_observation(self):
        for raw, drive, target in ((1000, 0, -97.0), (3000, 0, 97.0),
                                   (1000, 1, 97.0), (3000, 1, -97.0)):
            with self.subTest(raw=raw, drive=drive):
                self.bus.read_calls.clear()
                self.bus.raw["wrist_flex"] = raw
                self.bus.calibration["wrist_flex"].drive_mode = drive
                self.robot.calibration["wrist_flex"].drive_mode = drive
                self.mark_verified()
                callback = Mock(return_value=True)
                sent, observed = self.adapter.send_agent_targets(
                    {WRIST: target}, (WRIST,), {}, before_write=callback,
                    verify_calibrated_raw=True,
                )
                self.assert_one_raw_batch()
                callback.assert_called_once_with(observed)
                self.assertEqual(sent[WRIST], target)
                self.assertEqual(abs(observed[WRIST]), 100.0)
                self.assertTrue(1000 < self.bus.wire_writes[-1][4] < 3000)

    def test_nonselected_raw_guard_outside_calibration_blocks_the_other_axis_probe(self):
        callback = Mock(return_value=True)
        for raw in (999, 3001):
            with self.subTest(raw=raw):
                self.bus.read_calls.clear()
                self.bus.raw["wrist_flex"] = raw
                with self.assertRaises(CalibrationRangeError) as caught:
                    self.adapter.send_agent_targets(
                        {YAW: 2.0}, (YAW,), {}, before_write=callback,
                        verify_calibrated_raw=True, raw_guard_joints=(YAW, WRIST),
                    )
                self.assert_one_raw_batch()
                self.assertEqual(caught.exception.joints, (WRIST,))
                self.assertEqual(caught.exception.observation["raw_positions"][WRIST], raw)
                callback.assert_not_called()
                self.assertEqual(self.bus.write_calls, [])

    def test_legal_raw_guards_add_no_targets_and_invalid_guards_fail_before_io(self):
        self.bus.raw["wrist_flex"] = 1000
        for guards in ((WRIST,), [WRIST]):
            with self.subTest(guards=guards):
                self.bus.read_calls.clear()
                _, observed = self.adapter.send_agent_targets(
                    {YAW: 2.0}, (YAW,), {}, verify_calibrated_raw=True, raw_guard_joints=guards,
                )
                self.assert_one_raw_batch()
                self.assertEqual(observed[WRIST], -100.0)
                self.assertEqual(set(self.bus.write_calls[-1][1]), {"shoulder_pan"})
                self.assertEqual(set(self.bus.wire_writes[-1]), {1})
        self.bus.read_calls.clear()
        self.bus.write_calls.clear()
        invalid_inputs = (
            (WRIST, True), ({WRIST}, True), ({WRIST: 1}, True), (None, True),
            (("unknown.pos",), True), (([WRIST],), True), ((1,), True),
            ((WRIST, WRIST), True), ((WRIST,), False),
        )
        for guards, verify in invalid_inputs:
            with self.subTest(guards=guards, verify=verify):
                with self.assertRaises(ValueError):
                    self.adapter.send_agent_targets(
                        {YAW: 2.0}, (YAW,), {}, verify_calibrated_raw=verify, raw_guard_joints=guards,
                    )
                self.assertEqual(self.bus.read_calls, [])
                self.assertEqual(self.bus.write_calls, [])

    def test_clipped_inactive_axis_is_not_written_or_treated_as_a_fixed_hold(self):
        self.bus.raw["wrist_flex"] = 3001
        self.adapter.send_agent_targets({YAW: 2.0}, (YAW,), {}, verify_calibrated_raw=True)
        self.assert_one_raw_batch()
        self.assertEqual(set(self.bus.write_calls[-1][1]), {"shoulder_pan"})

    def test_raw_gated_send_keeps_callback_stop_and_relative_speed_cap(self):
        callback = Mock(return_value=False)
        self.adapter.send_agent_targets(
            {YAW: 90.0}, (YAW,), {}, before_write=callback, verify_calibrated_raw=True,
        )
        self.assertEqual(self.bus.write_calls, [])
        self.bus.read_calls.clear()
        sent, _ = self.adapter.send_agent_targets({YAW: 90.0}, (YAW,), {}, verify_calibrated_raw=True)
        cap = self.adapter.settings.max_relative_target
        cap = cap["shoulder_pan"] if isinstance(cap, dict) else cap
        self.assertEqual(sent[YAW], cap)
        self.assert_one_raw_batch()

    def test_unknown_selected_axis_is_rejected_before_raw_io(self):
        with self.assertRaisesRegex(ValueError, "unknown joint"):
            self.adapter.send_agent_targets({"unknown.pos": 0.0}, ("unknown.pos",), {},
                                            verify_calibrated_raw=True)
        self.assertEqual(self.bus.read_calls, [])
        self.assertEqual(self.bus.write_calls, [])

    def test_legacy_send_defaults_to_existing_observation_path(self):
        self.adapter._verified_calibration_fingerprint = None
        self.robot.get_observation = Mock(return_value=preview_initial_positions())
        self.adapter.send_agent_targets({YAW: 2.0}, (YAW,), {})
        self.robot.get_observation.assert_called_once_with()
        self.assertEqual(self.bus.read_calls, [])
        self.assertEqual(len(self.bus.write_calls), 1)

    def test_raw_hold_preserves_actual_outside_counts_instead_of_clipped_endpoints(self):
        self.bus.raw["wrist_flex"] = 999
        self.bus.raw["elbow_flex"] = 3001
        for drive in (0, 1):
            with self.subTest(drive=drive):
                self.bus.read_calls.clear()
                for name in self.bus.motors:
                    self.bus.calibration[name].drive_mode = drive
                    self.robot.calibration[name].drive_mode = drive
                self.mark_verified()
                observed = self.adapter.hold_raw_current()
                self.assert_one_raw_batch()
                register, goals, kwargs = self.bus.write_calls[-1]
                self.assertEqual(register, "Goal_Position")
                self.assertEqual(goals, self.bus.raw)
                self.assertEqual(kwargs, {"normalize": False, "num_retry": 0})
                self.assertEqual(self.bus.wire_writes[-1][4], 999)
                self.assertEqual(self.bus.wire_writes[-1][3], 3001)
                self.assertEqual(observed[WRIST], 100.0 if drive else -100.0)
                self.assertEqual(observed["elbow_flex.pos"], -100.0 if drive else 100.0)

    def test_raw_hold_read_failure_and_manual_positioning_never_write(self):
        self.bus.read_error = ConnectionError("simulated raw read failure")
        with self.assertRaisesRegex(RuntimeError, "simulated raw read failure"):
            self.adapter.hold_raw_current()
        self.assertEqual(self.bus.write_calls, [])
        self.bus.read_error = None
        self.bus.read_calls.clear()
        self.adapter.manual_range_active = True
        with self.assertRaisesRegex(RuntimeError, "manual positioning"):
            self.adapter.hold_raw_current()
        self.assertEqual(self.bus.read_calls, [])
        self.assertEqual(self.bus.write_calls, [])


if __name__ == "__main__":
    unittest.main()
