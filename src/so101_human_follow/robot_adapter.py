from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from .config import ALL_JOINTS, RobotSettings


LOGGER = logging.getLogger("so101_human_follow.control")


class CalibrationRangeError(ValueError):
    """An automatic command selected axes whose raw feedback is out of range."""

    def __init__(self, observation: dict[str, Any], joints: tuple[str, ...]):
        self.observation = observation
        self.joints = tuple(joints)
        details = "; ".join(
            f"{joint}: raw {observation['raw_positions'][joint]} outside "
            f"{observation['raw_limits'][joint]}" for joint in self.joints
        )
        super().__init__(
            "Refusing automatic movement or hold outside calibrated raw travel: " + details
        )


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
        self._verified_calibration_fingerprint: str | None = None
        self.manual_range_active = False
        self.fixed_positioning_active = False
        self._fixed_positioning_motors: tuple[str, ...] = ()

    def connect(self) -> dict[str, float]:
        self._verified_calibration_fingerprint = None
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
            self._verified_calibration_fingerprint = self.calibration_fingerprint()
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
            # LeRobot's calibrated-state check also depends on drive direction.
            # Include it in diagnostics so a drive-mode-only mismatch is actionable
            # instead of being reported as an unreadable generic mismatch.
            for field in ("drive_mode", "homing_offset", "range_min", "range_max"):
                saved_value = int(getattr(saved, field))
                actual_value = int(getattr(actual, field))
                if saved_value != actual_value:
                    differences.append(
                        f"{motor}.{field}: JSON={saved_value}, EEPROM={actual_value}"
                    )
        return differences

    def _cleanup_partial_connection(self) -> None:
        self._verified_calibration_fingerprint = None
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

    def calibration_ranges(self) -> dict[str, tuple[int, int]]:
        if self.robot is None:
            return {}
        return {
            f"{name}.pos": (int(calibration.range_min), int(calibration.range_max))
            for name, calibration in self.robot.calibration.items()
        }

    def read_calibration_observation(self) -> dict[str, Any]:
        """Read one raw position batch and normalize that same batch locally.

        Call only from the existing serial owner, like ``read_positions``. On
        Feetech, ``sync_read(..., normalize=False)`` retains the SDK's register
        decoding (STS3215 Present_Position has no signed conversion). Its homed
        Present_Position is the same coordinate used by
        LeRobot's raw range capture; do not apply homing or drive mode twice.
        ``bus._normalize`` applies the saved drive mode and clips raw readings
        to that range. ``clipped_joints`` therefore remains essential evidence:
        a clipped normalized endpoint is not an in-range encoder observation.
        Calibration metadata is cached and already verified at connection;
        this method never reads EEPROM, writes goals, or changes torque.
        """
        normalized_limits = self.normalized_joint_limits()
        robot = self.robot
        bus = robot.bus
        if getattr(bus, "apply_drive_mode", None) is not True:
            raise RuntimeError("Raw calibration observation requires Feetech drive-mode normalization.")
        fingerprint = self._verified_calibration_fingerprint
        raw_limits = self.calibration_ranges()
        names = [joint.removesuffix(".pos") for joint in ALL_JOINTS]
        ids = {name: int(bus.motors[name].id) for name in names}
        if len(set(ids.values())) != len(ALL_JOINTS):
            raise RuntimeError("Raw calibration observation requires six distinct motor IDs.")
        try:
            # One group-read attempt, without a second get_observation() sample.
            raw_values = bus.sync_read("Present_Position", names, normalize=False, num_retry=0)
        except Exception as exc:
            raise RuntimeError(f"Could not read raw calibration positions: {exc}") from exc
        if not isinstance(raw_values, Mapping) or set(raw_values) != set(names):
            raise RuntimeError("Raw calibration observation is missing the complete six-joint sample.")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_values.values()):
            raise RuntimeError("Raw calibration observation must contain integer encoder positions.")
        raw_positions = {f"{name}.pos": int(raw_values[name]) for name in names}
        try:
            normalized = bus._normalize({ids[name]: int(raw_values[name]) for name in names})
        except Exception as exc:
            raise RuntimeError(f"Could not normalize raw calibration positions: {exc}") from exc
        if not isinstance(normalized, Mapping) or set(normalized) != set(ids.values()):
            raise RuntimeError("Raw calibration normalization returned an incomplete six-joint sample.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in normalized.values()):
            raise RuntimeError("Raw calibration normalization returned a non-finite or invalid position.")
        # The I/O call may have yielded. Never combine a raw sample with a
        # different cached calibration, normalization mode, or motor bus.
        if (self.robot is not robot or robot.bus is not bus
                or getattr(bus, "apply_drive_mode", None) is not True
                or self._verified_calibration_fingerprint != fingerprint
                or self.normalized_joint_limits() != normalized_limits):
            raise RuntimeError("Motor calibration or normalization changed during the raw position read.")
        positions = {f"{name}.pos": float(normalized[ids[name]]) for name in names}
        in_range = {
            joint: raw_limits[joint][0] <= raw_positions[joint] <= raw_limits[joint][1]
            for joint in ALL_JOINTS
        }
        return {
            "positions": positions,
            "raw_positions": raw_positions,
            "raw_limits": raw_limits,
            "normalized_limits": normalized_limits,
            "in_range": in_range,
            "clipped_joints": tuple(joint for joint in ALL_JOINTS if not in_range[joint]),
            "calibration_fingerprint": fingerprint,
        }

    def normalized_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Read calibrated bus normalization bounds, without any serial I/O.

        Connection already checked EEPROM against saved calibration. Do not call
        ``is_calibrated`` here: LeRobot's Feetech property performs serial reads.
        These are calibrated motor travel bounds, not collision-free workspace
        limits; companion control must retain its smaller relative envelope.
        """
        if not self.connected or self.robot is None or self._verified_calibration_fingerprint is None:
            raise RuntimeError("Robot must be connected with verified calibration before reading motor limits.")
        bus = getattr(self.robot, "bus", None)
        if bus is None or getattr(bus, "is_connected", False) is not True:
            raise RuntimeError("The calibrated motor bus is not connected.")
        try:
            fingerprint = self.calibration_fingerprint()
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Motor calibration is incomplete or invalid.") from exc
        if fingerprint != self._verified_calibration_fingerprint:
            raise RuntimeError("Motor calibration changed after connection; reconnect before companion motion.")
        motors = getattr(bus, "motors", {})
        calibrated = getattr(bus, "calibration", {})
        saved = self.robot.calibration
        if not all(isinstance(value, Mapping) for value in (motors, calibrated, saved)):
            raise RuntimeError("Calibrated motor definitions are unavailable.")
        _make_lerobot_importable()
        from lerobot.motors import MotorNormMode

        result = {}
        for joint in ALL_JOINTS:
            motor_name = joint.removesuffix(".pos")
            motor = motors.get(motor_name)
            calibration = calibrated.get(motor_name)
            reference = saved.get(motor_name)
            if motor is None or calibration is None or reference is None:
                raise RuntimeError(f"Missing calibrated motor definition: {motor_name}.")
            try:
                fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
                current_values = tuple(int(getattr(calibration, name)) for name in fields)
                saved_values = tuple(int(getattr(reference, name)) for name in fields)
                low, high = float(calibration.range_min), float(calibration.range_max)
                if (current_values != saved_values or int(motor.id) != current_values[0]
                        or not math.isfinite(low + high) or not 0 <= low < high < (1 << 15)
                        or any(float(getattr(calibration, name)) != value for name, value in zip(fields, current_values))
                        or any(float(getattr(reference, name)) != value for name, value in zip(fields, saved_values))
                        or current_values[1] not in (0, 1)):
                    raise ValueError("Calibration is incomplete, changed, or has invalid raw travel.")
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"Invalid calibrated motor limits for {motor_name}: {exc}") from exc
            # Identity matches LeRobot _normalize/_unnormalize, which also use
            # enum identity. A similar string, degrees, or radians is unsupported.
            mode = getattr(motor, "norm_mode", None)
            if mode is MotorNormMode.RANGE_M100_100:
                result[joint] = (-100.0, 100.0)
            elif mode is MotorNormMode.RANGE_0_100:
                result[joint] = (0.0, 100.0)
            else:
                raise RuntimeError(f"Unsupported motor normalization for {motor_name}: {mode!r}.")
        return result

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

    def send_observed(
        self, action: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Read once, apply LeRobot's unchanged cap, then send an Agent goal.

        The measured positions precede this write; they confirm an earlier goal
        rather than pretending that a freshly sent goal has already been reached.
        Reusing that read avoids two serial round trips on each Agent tick.
        """
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active or self.fixed_positioning_active:
            raise RuntimeError("Motion commands are disabled during manual joint positioning.")
        if set(action) != set(ALL_JOINTS):
            raise ValueError("Agent action must contain all six joints.")
        goals = {joint.removesuffix(".pos"): float(action[joint]) for joint in ALL_JOINTS}
        if not all(math.isfinite(value) for value in goals.values()):
            raise ValueError("Agent action contains a non-finite target.")
        observed = self.read_positions()
        from lerobot.robots.utils import ensure_safe_goal_position

        goals = ensure_safe_goal_position(
            {motor: (value, observed[f"{motor}.pos"]) for motor, value in goals.items()},
            self.settings.max_relative_target,
        )
        self.robot.bus.sync_write("Goal_Position", goals)
        return {f"{motor}.pos": value for motor, value in goals.items()}, observed

    def disconnect(self) -> None:
        try:
            self._cleanup_partial_connection()
        finally:
            self.manual_range_active = False
            self.fixed_positioning_active = False
            self._fixed_positioning_motors = ()

    def send_agent_targets(
        self, action: dict[str, float], active_joints: tuple[str, ...],
        fixed_positions: dict[str, float],
        *, before_write: Callable[[dict[str, float]], bool] | None = None,
        verify_calibrated_raw: bool = False,
        raw_guard_joints: tuple[str, ...] = (),
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Write selected axes, optionally guarding additional axes in the raw sample.

        Raw guards can protect a multi-axis calibration while only one axis is
        probed. They are checked with the selected axes but never receive goals.
        """
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active or self.fixed_positioning_active:
            raise RuntimeError("Motion commands are disabled during manual joint positioning.")
        selected = set(active_joints) | set(fixed_positions)
        if len(set(active_joints)) != len(active_joints):
            raise ValueError("Agent listed one active joint more than once.")
        if set(active_joints) & set(fixed_positions):
            raise ValueError("An active Agent joint cannot also be held fixed.")
        if not selected <= set(ALL_JOINTS):
            raise ValueError("Agent requested an unknown joint.")
        if type(verify_calibrated_raw) is not bool:
            raise ValueError("verify_calibrated_raw must be a boolean.")
        if not isinstance(raw_guard_joints, (tuple, list)) or any(
            not isinstance(joint, str) for joint in raw_guard_joints
        ):
            raise ValueError("Raw guard joints must be a tuple or list of joint names.")
        guarded = set(raw_guard_joints)
        if len(guarded) != len(raw_guard_joints):
            raise ValueError("Raw guard listed one joint more than once.")
        if not guarded <= set(ALL_JOINTS):
            raise ValueError("Raw guard requested an unknown joint.")
        if guarded and not verify_calibrated_raw:
            raise ValueError("Raw guard joints require verify_calibrated_raw=True.")
        if verify_calibrated_raw:
            observation = self.read_calibration_observation()
            clipped = (selected | guarded) & set(observation["clipped_joints"])
            if clipped:
                raise CalibrationRangeError(
                    observation, tuple(joint for joint in ALL_JOINTS if joint in clipped),
                )
            observed = observation["positions"]
        else:
            observed = self.read_positions()
        goals = {
            joint.removesuffix(".pos"): float(fixed_positions[joint] if joint in fixed_positions else action[joint])
            for joint in selected
        }
        if not all(math.isfinite(value) for value in goals.values()):
            raise ValueError("Agent requested a non-finite target.")
        from lerobot.robots.utils import ensure_safe_goal_position
        caps = self.settings.max_relative_target
        if isinstance(caps, dict):
            caps = {motor: caps[motor] for motor in goals}
        goals = ensure_safe_goal_position(
            {motor: (goal, observed[f"{motor}.pos"]) for motor, goal in goals.items()}, caps,
        )
        if before_write is not None and not before_write(observed):
            return dict(observed), observed
        if goals:
            self.robot.bus.sync_write("Goal_Position", goals)
        sent = dict(observed)
        sent.update({f"{motor}.pos": goal for motor, goal in goals.items()})
        return sent, observed

    def hold_current(self) -> dict[str, float]:
        """Replace an interrupted goal with encoder positions, without torque changes."""
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active or self.fixed_positioning_active:
            raise RuntimeError("Cannot seed a powered hold during manual positioning.")
        observed = self.read_positions()
        self.robot.bus.sync_write(
            "Goal_Position", {joint.removesuffix(".pos"): value for joint, value in observed.items()}
        )
        return observed

    def hold_raw_current(self) -> dict[str, float]:
        """Hold the actual encoder pose even if normalization would clip it.

        Feetech Present_Position and Goal_Position use the same homed raw
        coordinate. ``normalize=False`` skips LeRobot's inverse normalization,
        so the SDK writes the sampled encoder values rather than replacing an
        out-of-recording-range pose with a calibrated endpoint. Its ordinary
        register encoding still applies. This is a hold, not an automatic
        return into the recorded range, and never changes torque/calibration.
        """
        if not self.connected or self.robot is None:
            raise RuntimeError("Robot is not connected.")
        if self.manual_range_active or self.fixed_positioning_active:
            raise RuntimeError("Cannot seed a powered hold during manual positioning.")
        observation = self.read_calibration_observation()
        self.robot.bus.sync_write(
            "Goal_Position",
            {joint.removesuffix(".pos"): observation["raw_positions"][joint] for joint in ALL_JOINTS},
            normalize=False, num_retry=0,
        )
        return dict(observation["positions"])


class SO101CalibrationSession:
    """Manual six-axis LeRobot calibration without ever enabling motor torque."""

    def __init__(
        self,
        settings: RobotSettings,
        *,
        device_factory: Callable[[], Any] | None = None,
        single_turn_wrist: bool = True,
    ):
        self.settings = settings
        self._device_factory = device_factory or self._make_follower
        self._single_turn_wrist = single_turn_wrist
        self.robot: Any | None = None
        self.active = False
        self.homing_offsets: dict[str, int] | None = None

    def _make_follower(self) -> Any:
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
            disable_torque_on_disconnect=True,
        )
        return SO101Follower(config)

    def start(self) -> dict[str, int]:
        if self.active or self.robot is not None:
            raise RuntimeError("Base calibration is already active.")
        self.robot = self._device_factory()
        try:
            from lerobot.motors.feetech import OperatingMode

            # Open the bus directly so device configuration cannot enable torque.
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
            if self._single_turn_wrist and motor == "wrist_roll" and high > 4095:
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

