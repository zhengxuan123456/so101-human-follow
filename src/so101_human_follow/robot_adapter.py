from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from .config import ALL_JOINTS, RobotSettings


LOGGER = logging.getLogger("so101_human_follow.control")


def _make_lerobot_importable() -> None:
    try:
        import lerobot  # noqa: F401

        return
    except ImportError:
        pass

    embedding_root = Path(__file__).resolve().parents[3]
    source = embedding_root / "lerobot" / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))


class SO101RobotAdapter:
    def __init__(self, settings: RobotSettings):
        self.settings = settings
        self.robot: Any | None = None
        self.connected = False
        self.manual_range_active = False
        self.fixed_positioning_active = False
        self._fixed_positioning_motors: tuple[str, ...] = ()

    def connect(self) -> dict[str, float]:
        _make_lerobot_importable()
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=self.settings.port,
            id=self.settings.id,
            cameras={},
            max_relative_target=(
                dict(self.settings.max_relative_target)
                if isinstance(self.settings.max_relative_target, dict)
                else float(self.settings.max_relative_target)
            ),
            disable_torque_on_disconnect=self.settings.disable_torque_on_disconnect,
        )
        self.robot = SO101Follower(config)
        if not self.robot.calibration:
            raise RuntimeError(
                f"No calibration was loaded for SO-101 id '{self.settings.id}'. "
                "This program will not start automatic calibration."
            )

        try:
            self.robot.connect(calibrate=False)
            self.connected = True
            if not self.robot.is_calibrated:
                differences = self._calibration_differences()
                detail = "; ".join(differences) if differences else "no field-level difference was readable"
                LOGGER.error("motor calibration mismatch | differences=%s", detail)
                raise RuntimeError(
                    "Motor calibration does not match the saved JSON. "
                    f"Differences: {detail}. Refusing motion; run calibration again."
                )
            return self.read_positions()
        except Exception:
            self._cleanup_partial_connection()
            raise

    def _calibration_differences(self) -> list[str]:
        if self.robot is None:
            return ["robot is unavailable"]
        try:
            motor_calibration = self.robot.bus.read_calibration()
        except Exception as exc:
            return [f"EEPROM read failed: {exc}"]

        saved_calibration = self.robot.calibration
        differences: list[str] = []
        for motor in sorted(set(saved_calibration) | set(motor_calibration)):
            saved = saved_calibration.get(motor)
            actual = motor_calibration.get(motor)
            if saved is None:
                differences.append(f"{motor}: missing from JSON")
                continue
            if actual is None:
                differences.append(f"{motor}: missing from EEPROM readback")
                continue
            for field in ("homing_offset", "range_min", "range_max"):
                saved_value = int(getattr(saved, field))
                actual_value = int(getattr(actual, field))
                if saved_value != actual_value:
                    differences.append(
                        f"{motor}.{field}: JSON={saved_value}, EEPROM={actual_value}"
                    )
        return differences

    def _cleanup_partial_connection(self) -> None:
        if self.robot is None:
            self.connected = False
            return
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
                self.connected = False
                return
        except Exception as exc:
            print(f"Robot-level cleanup warning: {exc}", file=sys.stderr)

        bus = getattr(self.robot, "bus", None)
        try:
            if bus is not None and bus.is_connected:
                bus.disconnect(self.settings.disable_torque_on_disconnect)
        except Exception as exc:
            print(f"Motor-bus cleanup warning: {exc}", file=sys.stderr)
        self.connected = False

    def read_positions(self) -> dict[str, float]:
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        observation = self.robot.get_observation()
        positions = {joint: float(observation[joint]) for joint in ALL_JOINTS}
        if not all(math.isfinite(value) for value in positions.values()):
            raise RuntimeError("Robot returned a non-finite joint position.")
        return positions

    def calibration_fingerprint(self) -> str:
        if self.robot is None or not self.robot.calibration:
            raise RuntimeError("Robot calibration is not available.")
        payload = {
            name: {
                "id": int(calibration.id),
                "drive_mode": int(calibration.drive_mode),
                "homing_offset": int(calibration.homing_offset),
                "range_min": int(calibration.range_min),
                "range_max": int(calibration.range_max),
            }
            for name, calibration in sorted(self.robot.calibration.items())
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _motor_names(joints: tuple[str, ...] | list[str]) -> list[str]:
        return [joint.removesuffix(".pos") for joint in joints]

    def begin_manual_range(self, joints: tuple[str, ...] | list[str]) -> dict[str, float]:
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.fixed_positioning_active:
            raise RuntimeError("Finish ID1-ID2 positioning before safe-range capture.")
        if self.manual_range_active:
            raise RuntimeError("Manual safe-range capture is already active.")
        positions = self.read_positions()
        motors = self._motor_names(joints)
        try:
            self.robot.bus.disable_torque(motors)
        except Exception:
            # A looped bus write may have disabled only a prefix of the motors.
            # Establish a current-position goal and re-enable the full selection.
            goals = {motor: positions[f"{motor}.pos"] for motor in motors}
            try:
                self.robot.bus.sync_write("Goal_Position", goals)
                self.robot.bus.enable_torque(motors)
            finally:
                self.manual_range_active = False
            raise
        self.manual_range_active = True
        return positions

    def finish_manual_range(
        self,
        joints: tuple[str, ...] | list[str],
        *,
        positions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Hold the present pose before re-enabling the manually released joints."""

        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if not self.manual_range_active:
            raise RuntimeError("Manual safe-range capture is not active.")
        motors = self._motor_names(joints)
        try:
            if positions is None:
                positions = self.read_positions()
            else:
                positions = {joint: float(positions[joint]) for joint in ALL_JOINTS}
                if not all(math.isfinite(value) for value in positions.values()):
                    raise RuntimeError("The proposed lock pose contains a non-finite position.")
            goals = {motor: positions[f"{motor}.pos"] for motor in motors}
            # Never re-enable torque against an old Goal_Position: that can cause a jump.
            self.robot.bus.sync_write("Goal_Position", goals)
            self.robot.bus.enable_torque(motors)
        except Exception:
            # Keep these joints torque-free if a safe current-position goal could not be
            # established. The operator must support the arm and use the power switch.
            raise
        else:
            self.manual_range_active = False
            return positions

    def begin_fixed_positioning(
        self, joints: tuple[str, ...] | list[str]
    ) -> dict[str, float]:
        """Release only the selected fixed joints for manual positioning."""

        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active:
            raise RuntimeError("Finish safe-range capture before positioning ID1-ID2.")
        if self.fixed_positioning_active:
            raise RuntimeError("ID1-ID2 manual positioning is already active.")

        positions = self.read_positions()
        motors = tuple(self._motor_names(joints))
        if not motors:
            raise RuntimeError("No fixed joints were selected for manual positioning.")

        # Mark the operation active before the looped torque write.  If that write is
        # only partially successful and recovery also fails, motion commands stay
        # blocked and the operator can safely use the physical power switch.
        self.fixed_positioning_active = True
        self._fixed_positioning_motors = motors
        try:
            self.robot.bus.disable_torque(list(motors))
        except Exception as disable_exc:
            goals = {motor: positions[f"{motor}.pos"] for motor in motors}
            try:
                self.robot.bus.sync_write("Goal_Position", goals)
                self.robot.bus.enable_torque(list(motors))
            except Exception as restore_exc:
                raise RuntimeError(
                    "Could not safely restore ID1-ID2 torque after a partial release. "
                    "Support the arm and use the physical power switch."
                ) from restore_exc
            else:
                self.fixed_positioning_active = False
                self._fixed_positioning_motors = ()
            raise disable_exc
        return positions

    def finish_fixed_positioning(
        self, joints: tuple[str, ...] | list[str]
    ) -> dict[str, float]:
        """Lock the selected fixed joints at their present manually chosen pose."""

        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if not self.fixed_positioning_active:
            raise RuntimeError("ID1-ID2 manual positioning is not active.")

        motors = tuple(self._motor_names(joints))
        if motors != self._fixed_positioning_motors:
            raise RuntimeError(
                "The fixed-joint selection changed while ID1-ID2 torque was released."
            )

        try:
            positions = self.read_positions()
            goals = {motor: positions[f"{motor}.pos"] for motor in motors}
            # Seed the present pose before torque is restored.  Enabling against an
            # older Goal_Position can make the arm jump unexpectedly.
            self.robot.bus.sync_write("Goal_Position", goals)
            self.robot.bus.enable_torque(list(motors))
        except Exception:
            # Preserve the active flag so send() remains blocked while any selected
            # joint may still be torque-free.
            raise
        else:
            self.fixed_positioning_active = False
            self._fixed_positioning_motors = ()
            return positions

    def send(self, action: dict[str, float]) -> dict[str, float]:
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active or self.fixed_positioning_active:
            raise RuntimeError("Motion commands are disabled during manual joint positioning.")
        return self.robot.send_action(action)

    def disconnect(self) -> None:
        try:
            self._cleanup_partial_connection()
        finally:
            self.manual_range_active = False
            self.fixed_positioning_active = False
            self._fixed_positioning_motors = ()


class SO101CalibrationSession:
    """Manual six-axis LeRobot calibration without ever enabling motor torque."""

    def __init__(self, settings: RobotSettings):
        self.settings = settings
        self.robot: Any | None = None
        self.active = False
        self.homing_offsets: dict[str, int] | None = None

    def start(self) -> dict[str, int]:
        if self.active or self.robot is not None:
            raise RuntimeError("Base calibration is already active.")
        _make_lerobot_importable()
        from lerobot.motors.feetech import OperatingMode
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=self.settings.port,
            id=self.settings.id,
            cameras={},
            max_relative_target=(
                dict(self.settings.max_relative_target)
                if isinstance(self.settings.max_relative_target, dict)
                else float(self.settings.max_relative_target)
            ),
            disable_torque_on_disconnect=True,
        )
        self.robot = SO101Follower(config)
        try:
            # Deliberately bypass SO101Follower.connect(): it calls configure(), whose
            # context manager re-enables torque. Calibration must remain hand-movable.
            self.robot.bus.connect()
            self.robot.bus.disable_torque()
            for motor in self.robot.bus.motors:
                self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.active = True
            return self.read_raw_positions()
        except Exception:
            self._cleanup()
            raise

    def read_raw_positions(self) -> dict[str, int]:
        if not self.active or self.robot is None:
            raise RuntimeError("Base calibration is not active.")
        positions = self.robot.bus.sync_read("Present_Position", normalize=False)
        expected = set(self.robot.bus.motors)
        if set(positions) != expected:
            raise RuntimeError("Motor bus returned an incomplete raw-position record.")
        result = {str(motor): int(value) for motor, value in positions.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError("Motor bus returned a non-finite raw position.")
        return result

    def capture_middle(self) -> tuple[dict[str, int], dict[str, int]]:
        if not self.active or self.robot is None:
            raise RuntimeError("Base calibration is not active.")
        offsets = self.robot.bus.set_half_turn_homings()
        self.homing_offsets = {str(motor): int(value) for motor, value in offsets.items()}
        if set(self.homing_offsets) != set(self.robot.bus.motors):
            raise RuntimeError("Homing-offset capture did not return all six motors.")
        return dict(self.homing_offsets), self.read_raw_positions()

    def finish(
        self,
        range_mins: dict[str, int],
        range_maxes: dict[str, int],
    ) -> dict[str, Any]:
        if not self.active or self.robot is None:
            raise RuntimeError("Base calibration is not active.")
        if self.homing_offsets is None:
            raise RuntimeError("Capture the six-axis middle position before saving calibration.")
        motors = set(self.robot.bus.motors)
        if set(range_mins) != motors or set(range_maxes) != motors:
            raise ValueError("Calibration ranges must contain all six motors.")

        checked_mins: dict[str, int] = {}
        checked_maxes: dict[str, int] = {}
        for motor in self.robot.bus.motors:
            low = int(range_mins[motor])
            high = int(range_maxes[motor])
            if low < 0 or low >= (1 << 15):
                raise ValueError(
                    f"Motor '{motor}' range_min={low} is not a valid Feetech raw position "
                    "(bit 15 indicates a wrapped/negative value). Re-run calibration without crossing a turn."
                )
            if high < 0 or high >= (1 << 15):
                raise ValueError(
                    f"Motor '{motor}' range_max={high} is not a valid Feetech raw position "
                    "(bit 15 indicates a wrapped/negative value). Re-run calibration without crossing a turn."
                )
            if low >= high:
                raise ValueError(f"Motor '{motor}' did not move through a valid range.")
            if motor == "wrist_roll" and high > 4095:
                raise ValueError(
                    f"ID5 wrist_roll 最大值超过4095（当前为 {high}），"
                    "请重新采集且不要跨圈，并注意摄像头线缆。"
                )
            checked_mins[motor] = low
            checked_maxes[motor] = high

        _make_lerobot_importable()
        from lerobot.motors import MotorCalibration

        calibration = {
            motor: MotorCalibration(
                id=motor_config.id,
                drive_mode=0,
                homing_offset=self.homing_offsets[motor],
                range_min=checked_mins[motor],
                range_max=checked_maxes[motor],
            )
            for motor, motor_config in self.robot.bus.motors.items()
        }
        calibration_path = str(self.robot.calibration_fpath)
        try:
            # Torque has remained disabled since start(). These are EEPROM writes;
            # never wrap them in torque_disabled(), which would enable torque on exit.
            self.robot.calibration = calibration
            self.robot.bus.write_calibration(calibration)
            readback = self.robot.bus.read_calibration()
            differences: list[str] = []
            for motor, expected in calibration.items():
                actual = readback.get(motor)
                if actual is None:
                    differences.append(f"{motor}: missing from EEPROM readback")
                    continue
                for field in ("homing_offset", "range_min", "range_max"):
                    requested_value = int(getattr(expected, field))
                    actual_value = int(getattr(actual, field))
                    if requested_value != actual_value:
                        differences.append(
                            f"{motor}.{field}: requested={requested_value}, EEPROM={actual_value}"
                        )
            unexpected = sorted(set(readback) - set(calibration))
            if unexpected:
                differences.append(f"unexpected EEPROM motors: {', '.join(unexpected)}")
            if differences:
                details = "; ".join(differences)
                raise RuntimeError(
                    "Calibration EEPROM readback did not match the requested values; "
                    f"calibration JSON was not saved. {details}"
                )
            self.robot._save_calibration()
            final_raw = self._seed_current_raw_goal()
            return {
                "calibration_path": calibration_path,
                "final_raw_positions": final_raw,
            }
        finally:
            self._cleanup()

    def cancel(self) -> None:
        if self.robot is None:
            self.active = False
            self.homing_offsets = None
            return
        try:
            if self.active:
                try:
                    self._seed_current_raw_goal()
                except Exception as exc:
                    print(f"Calibration cancel goal-seed warning: {exc}", file=sys.stderr)
        finally:
            self._cleanup()

    def _seed_current_raw_goal(self) -> dict[str, int]:
        if not self.active or self.robot is None:
            raise RuntimeError("Base calibration is not active.")
        positions = self.read_raw_positions()
        self.robot.bus.sync_write("Goal_Position", positions, normalize=False)
        return positions

    def _cleanup(self) -> None:
        robot = self.robot
        try:
            bus = None if robot is None else getattr(robot, "bus", None)
            if bus is not None and bus.is_connected:
                # This never enables torque; disconnect(True) explicitly disables it.
                bus.disconnect(disable_torque=True)
        except Exception as exc:
            print(f"Calibration bus cleanup warning: {exc}", file=sys.stderr)
        finally:
            self.active = False
            self.homing_offsets = None
            self.robot = None


def preview_initial_positions() -> dict[str, float]:
    return {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }

