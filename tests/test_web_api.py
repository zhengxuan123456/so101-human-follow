import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

from so101_human_follow.config import load_config
from so101_human_follow.pose_math import PoseFeatures
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.web_app import create_app
from so101_human_follow.web_engine import (
    NEUTRAL_CAPTURE_TIMEOUT_S,
    ControlCommand,
    ControlEngine,
    ControlError,
)


class FakeEngine:
    def __init__(self):
        self.robot_allowed = True
        self.config = SimpleNamespace(
            robot=SimpleNamespace(port="COM4", id="my_awesome_follower_arm")
        )
        self.owner = None
        self.submissions = []
        self.disarms = []

    def heartbeat(self, client_id):
        if self.owner is None:
            self.owner = client_id
        return self.owner == client_id

    def status(self, client_id):
        return {
            "operator": self.owner == client_id,
            "port": "COM4",
            "robot_id": "my_awesome_follower_arm",
        }

    def submit(self, name, payload=None, *, client_id=None):
        self.submissions.append((name, payload or {}, client_id))
        return {"ok": True, "name": name}

    def disarm(self, reason="operator request"):
        self.disarms.append(reason)

    def stop_following(self, reason="operator pressed stop-following"):
        self.disarm(reason)

    def wait_for_frame(self, after_revision, timeout_s=2.0):
        return after_revision, None


class WebApiTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.app = create_app(self.engine, control_token="test-control-token")
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self.client.get("/")
        self.headers = {
            "X-Client-Id": "client_A_123456789",
            "Origin": "http://localhost",
        }

    def test_root_sets_http_only_same_site_cookie(self):
        fresh = self.app.test_client()
        response = fresh.get("/")
        cookie = response.headers["Set-Cookie"]
        self.assertIn("so101_control=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

    def test_api_requires_page_cookie(self):
        fresh = self.app.test_client()
        response = fresh.get(
            "/api/status", headers={"X-Client-Id": "client_B_123456789"}
        )
        self.assertEqual(response.status_code, 403)

    def test_non_loopback_host_header_is_rejected(self):
        response = self.client.get("/", headers={"Host": "example.invalid"})
        self.assertEqual(response.status_code, 403)

    def test_only_one_browser_gets_control_lease(self):
        first = self.client.get("/api/status", headers=self.headers)
        self.assertTrue(first.get_json()["operator"])

        other = self.app.test_client()
        other.get("/")
        other_headers = {
            "X-Client-Id": "client_B_123456789",
            "Origin": "http://localhost",
        }
        status = other.get("/api/status", headers=other_headers)
        self.assertFalse(status.get_json()["operator"])
        rejected = other.post("/api/disarm", headers=other_headers)
        self.assertEqual(rejected.status_code, 409)

    def test_post_requires_same_origin(self):
        self.client.get("/api/status", headers=self.headers)
        response = self.client.post(
            "/api/disarm", headers={"X-Client-Id": self.headers["X-Client-Id"]}
        )
        self.assertEqual(response.status_code, 403)

    def test_commands_are_forwarded_as_high_level_intents(self):
        self.client.get("/api/status", headers=self.headers)
        connect = self.client.post(
            "/api/connect",
            json={"confirmation": "123"},
            headers=self.headers,
        )
        neutral = self.client.post("/api/neutral", headers=self.headers)
        arm = self.client.post(
            "/api/arm",
            json={"confirmation": "123"},
            headers=self.headers,
        )
        self.assertEqual(connect.status_code, 200)
        self.assertEqual(neutral.status_code, 200)
        self.assertEqual(arm.status_code, 200)
        self.assertEqual(
            self.engine.submissions,
            [
                ("connect", {"confirmation": "123"}, "client_A_123456789"),
                ("collect_neutral", {}, "client_A_123456789"),
                ("arm", {"confirmation": "123"}, "client_A_123456789"),
            ],
        )

    def test_stop_is_immediate_and_does_not_release_torque(self):
        self.client.get("/api/status", headers=self.headers)
        response = self.client.post("/api/disarm", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.engine.disarms, ["operator pressed stop-following"])
        self.assertEqual(self.engine.submissions, [])
        self.assertIn("did not request torque release", response.get_json()["message"])

    def test_safe_range_commands_are_forwarded_as_high_level_intents(self):
        self.client.get("/api/status", headers=self.headers)
        responses = [
            self.client.post(
                "/api/safe-range/fixed",
                json={"confirmation": "123"},
                headers=self.headers,
            ),
            self.client.post(
                "/api/safe-range/start",
                json={"confirmation": "123"},
                headers=self.headers,
            ),
            self.client.post(
                "/api/safe-range/finish",
                json={"confirmation": "123"},
                headers=self.headers,
            ),
            self.client.post("/api/safe-range/cancel", headers=self.headers),
        ]
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(
            self.engine.submissions,
            [
                ("capture_fixed", {"confirmation": "123"}, "client_A_123456789"),
                ("start_safe_range", {"confirmation": "123"}, "client_A_123456789"),
                (
                    "finish_safe_range",
                    {"confirmation": "123"},
                    "client_A_123456789",
                ),
                ("cancel_safe_range", {}, "client_A_123456789"),
            ],
        )


class WebEngineSnapshotTest(unittest.TestCase):
    def test_snapshot_marks_lower_two_fixed_and_upper_four_moving(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_snapshot_123456"
        engine.heartbeat(client_id)
        status = engine.status(client_id)
        roles = [(row["id"], row["name"], row["role"]) for row in status["joints"]]
        self.assertEqual(
            roles,
            [
                (1, "shoulder_pan.pos", "fixed"),
                (2, "shoulder_lift.pos", "fixed"),
                (3, "elbow_flex.pos", "moving"),
                (4, "wrist_flex.pos", "moving"),
                (5, "wrist_roll.pos", "moving"),
                (6, "gripper.pos", "moving"),
            ],
        )

    def test_neutral_channels_can_be_collected_on_separate_frames(self):
        config = load_config("config.json")
        config = replace(config, pose=replace(config.pose, neutral_sample_count=1))
        engine = ControlEngine(config, robot_allowed=False)
        engine._collecting_neutral = True
        features = PoseFeatures(
            pan=0.0,
            lift=0.2,
            elbow=0.4,
            center_x=0.5,
            hand_roll=-0.3,
            hand_open=0.7,
        )
        now = time.monotonic()
        for offset, feature in enumerate(("lift", "elbow", "hand_roll", "hand_open")):
            engine._update_tracking(
                PoseDetection(
                    features,
                    (),
                    0,
                    "test channel",
                    valid_features=frozenset({feature}),
                ),
                now + offset * 0.01,
            )
        engine._neutral_started_at = time.monotonic() - NEUTRAL_CAPTURE_TIMEOUT_S - 0.1
        engine._update_tracking(
            PoseDetection(
                features,
                (),
                0,
                "test channel",
                valid_features=frozenset({"hand_open"}),
            ),
            time.monotonic(),
        )
        self.assertFalse(engine._collecting_neutral)
        self.assertIsNotNone(engine._neutral)
        self.assertAlmostEqual(engine._neutral.lift, features.lift)
        self.assertAlmostEqual(engine._neutral.elbow, features.elbow)
        self.assertAlmostEqual(engine._neutral.hand_roll, features.hand_roll)
        self.assertAlmostEqual(engine._neutral.hand_open, features.hand_open)

    def test_one_valid_channel_keeps_tracking_alive_until_all_are_lost(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        engine._armed = True
        engine._motion_stop.clear()
        now = time.monotonic()
        features = PoseFeatures(0.0, 0.2, 0.5, 0.5, 0.0, 0.5)
        engine._update_tracking(
            PoseDetection(
                features,
                (),
                0,
                "lift only",
                valid_features=frozenset({"lift"}),
            ),
            now,
        )
        engine._update_tracking(
            PoseDetection(None, (), 0, "no channels"),
            now + config.safety.tracking_timeout_s / 2.0,
        )
        self.assertTrue(engine._armed)
        engine._update_tracking(
            PoseDetection(None, (), 0, "no channels"),
            now + config.safety.tracking_timeout_s + 0.01,
        )
        self.assertFalse(engine._armed)

    def test_old_arm_command_is_invalidated_by_disarm(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_epoch_123456789"
        engine.heartbeat(client_id)
        command = ControlCommand(
            "arm",
            {"confirmation": "123"},
            client_id=client_id,
            submitted_epoch=engine._epoch,
        )
        engine.disarm("test stop")
        with self.assertRaisesRegex(ControlError, "cancelled"):
            engine._authorize_command(command)

    def test_old_safe_range_start_is_invalidated_by_disarm(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_range_epoch_12345"
        engine.heartbeat(client_id)
        command = ControlCommand(
            "start_safe_range",
            {"confirmation": "123"},
            client_id=client_id,
            submitted_epoch=engine._epoch,
        )
        engine.disarm("page closed")
        with self.assertRaisesRegex(ControlError, "cancelled"):
            engine._authorize_command(command)

    def test_disarm_requests_active_safe_range_cancellation(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        engine._range_state = "RECORDING"
        engine.disarm("page closed")
        self.assertIn("page closed", engine._range_cancel_requested)

    def test_arm_rechecks_epoch_after_initial_authorization(self):
        class ConnectedRobot:
            connected = True

        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_atomic_arm_12345"
        engine.heartbeat(client_id)
        features = PoseFeatures(0.0, 0.0, 0.5, 0.5)
        engine._robot = ConnectedRobot()
        engine._robot_state = "CONNECTED_DISARMED"
        engine._neutral = features
        engine._latest_features = features
        engine._last_pose_time = time.monotonic()
        engine._valid_streak = config.safety.valid_frames_before_arm
        command = ControlCommand(
            "arm",
            {"confirmation": "123"},
            client_id=client_id,
            submitted_epoch=engine._epoch,
        )

        engine._authorize_command(command)
        engine.disarm("stop after initial authorization")
        with self.assertRaisesRegex(ControlError, "cancelled"):
            engine._execute_command(command)
        self.assertFalse(engine._armed)

    def test_stop_refuses_to_promise_hold_when_release_is_pending(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        engine._armed = True
        engine._motion_stop.clear()
        engine._release_pending = True
        with self.assertRaisesRegex(ControlError, "Following was DISARMED"):
            engine.stop_following()
        self.assertFalse(engine._armed)
        self.assertTrue(engine._motion_stop.is_set())

    def test_hardware_commands_require_exact_confirmation_code(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        with self.assertRaisesRegex(ControlError, "123"):
            engine._connect_robot("connect COM4")
        with self.assertRaisesRegex(ControlError, "123"):
            engine._arm("ARM")
        with self.assertRaisesRegex(ControlError, "123"):
            engine._disconnect_robot("DISCONNECT COM4")

    def test_worker_rechecks_lease_before_command_execution(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_lease_12345678"
        engine.heartbeat(client_id)
        command = ControlCommand(
            "connect",
            {"confirmation": "123"},
            client_id=client_id,
            submitted_epoch=engine._epoch,
        )
        engine._operator_heartbeat -= engine.lease_timeout_s + 1.0
        with self.assertRaisesRegex(ControlError, "lease expired"):
            engine._authorize_command(command)

    def test_expired_lease_sets_motion_stop_before_full_disarm(self):
        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        engine._operator_client = "old_client_12345678"
        engine._operator_heartbeat = time.monotonic() - engine.lease_timeout_s - 1.0
        engine._armed = True
        engine._motion_stop.clear()
        observed = []
        original_disarm = engine.disarm

        def observe_stop(reason):
            observed.append(engine._motion_stop.is_set())
            original_disarm(reason)

        engine.disarm = observe_stop
        self.assertTrue(engine.heartbeat("new_client_12345678"))
        self.assertEqual(observed, [True])
        self.assertFalse(engine._armed)

    def test_queued_torque_release_disarms_before_waiting_for_worker(self):
        class AliveThread:
            @staticmethod
            def is_alive():
                return True

        class ConnectedRobot:
            connected = True

        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_release_123456"
        engine.heartbeat(client_id)
        engine._thread = AliveThread()
        engine._robot = ConnectedRobot()
        engine._robot_state = "ARMED"
        engine._armed = True
        engine._motion_stop.clear()

        with self.assertRaisesRegex(ControlError, "cancelled before it started"):
            engine.submit(
                "disconnect",
                {"confirmation": "123"},
                client_id=client_id,
                timeout_s=0.0,
            )
        self.assertFalse(engine._armed)
        self.assertTrue(engine._motion_stop.is_set())

    def test_disarm_waits_for_inflight_write_then_blocks_future_writes(self):
        class BlockingRobot:
            def __init__(self):
                self.connected = True
                self.started = threading.Event()
                self.release = threading.Event()
                self.send_count = 0

            def send(self, action):
                self.send_count += 1
                self.started.set()
                if not self.release.wait(2.0):
                    raise RuntimeError("test timed out waiting to release send")
                return dict(action)

        config = load_config("config.json")
        engine = ControlEngine(config, robot_allowed=True)
        robot = BlockingRobot()
        neutral = PoseFeatures(0.0, 0.0, 0.5, 0.5)
        features = PoseFeatures(0.2, 0.2, 0.7, 0.5)
        engine._robot = robot
        engine._neutral = neutral
        engine._armed = True
        engine._motion_stop.clear()
        engine._valid_streak = config.safety.valid_frames_before_arm

        errors = []

        def send_once():
            try:
                engine._send_if_due(features, time.monotonic(), 0.0)
            except Exception as exc:  # pragma: no cover - assertion reports details
                errors.append(exc)

        send_thread = threading.Thread(target=send_once)
        send_thread.start()
        self.assertTrue(robot.started.wait(1.0))

        disarm_done = threading.Event()

        def stop():
            engine.disarm("test stop")
            disarm_done.set()

        stop_thread = threading.Thread(target=stop)
        stop_thread.start()
        self.assertFalse(disarm_done.wait(0.05))
        robot.release.set()
        send_thread.join(1.0)
        stop_thread.join(1.0)

        self.assertEqual(errors, [])
        self.assertTrue(disarm_done.is_set())
        self.assertFalse(engine._armed)
        engine._send_if_due(features, time.monotonic() + 1.0, 0.0)
        self.assertEqual(robot.send_count, 1)


if __name__ == "__main__":
    unittest.main()
