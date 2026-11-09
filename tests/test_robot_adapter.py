import unittest
from types import SimpleNamespace

from so101_human_follow.config import load_config
from so101_human_follow.robot_adapter import SO101RobotAdapter


class FakeBus:
    def __init__(self):
        self.is_connected = True
        self.disconnect_calls = 0

    def disconnect(self, disable_torque):
        self.disconnect_calls += 1
        self.is_connected = False


class PartialRobot:
    is_connected = False

    def __init__(self):
        self.bus = FakeBus()


class ConnectedRobot:
    is_connected = True

    def __init__(self):
        self.disconnect_calls = 0

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False


class RobotAdapterCleanupTest(unittest.TestCase):
    def setUp(self):
        settings = load_config("config.json").robot
        self.adapter = SO101RobotAdapter(settings)

    def test_partial_bus_connection_is_cleaned(self):
        robot = PartialRobot()
        self.adapter.robot = robot
        self.adapter._cleanup_partial_connection()
        self.assertEqual(robot.bus.disconnect_calls, 1)
        self.assertFalse(self.adapter.connected)

    def test_connected_robot_uses_robot_disconnect(self):
        robot = ConnectedRobot()
        self.adapter.robot = robot
        self.adapter.connected = True
        self.adapter.disconnect()
        self.assertEqual(robot.disconnect_calls, 1)
        self.assertFalse(self.adapter.connected)

    def test_calibration_differences_report_drive_mode(self):
        saved = SimpleNamespace(
            drive_mode=0,
            homing_offset=100,
            range_min=200,
            range_max=300,
        )
        actual = SimpleNamespace(
            drive_mode=1,
            homing_offset=100,
            range_min=200,
            range_max=300,
        )
        bus = SimpleNamespace(read_calibration=lambda: {"elbow_flex": actual})
        self.adapter.robot = SimpleNamespace(
            bus=bus,
            calibration={"elbow_flex": saved},
        )

        differences = self.adapter._calibration_differences()

        self.assertEqual(
            differences,
            ["elbow_flex.drive_mode: JSON=0, EEPROM=1"],
        )


if __name__ == "__main__":
    unittest.main()
