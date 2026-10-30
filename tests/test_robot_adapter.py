import unittest

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


if __name__ == "__main__":
    unittest.main()
