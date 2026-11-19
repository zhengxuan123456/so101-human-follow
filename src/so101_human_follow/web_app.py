from __future__ import annotations

import argparse
import copy
import ipaddress
import logging
import math
import re
import secrets
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, make_response, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from .agent_actions import MAX_LOCAL_AGENT_ACTIONS
from .agent_schema import InteractionPlan
from .agent_service import AgentService
from .camera_device import list_camera_names
from .config import load_config
from .speech_normalizer import normalize_speech
from .simulation import ActionLibrary, DeletedActionError, create_simulation_blueprint
from .speech_synthesis import (
    SpeechBusyError,
    SpeechSynthesisError,
    SpeechSynthesizer,
    SpeechTimeoutError,
    SpeechUnavailableError,
)
from .web_engine import AGENT_INTERACTION_PREEMPT_REASON, CONTROL_CONFIRMATION_CODE, ControlEngine, ControlError


COOKIE_NAME = "so101_control"
CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,96}$")
LOGGER = logging.getLogger(__name__)


def configure_backend_logging(project_dir: Path) -> Path:
    """Write durable backend diagnostics without exposing them in the web UI."""

    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "control.log"
    package_logger = logging.getLogger("so101_human_follow")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(getattr(handler, "name", "") == "so101_file" for handler in package_logger.handlers):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.name = "so101_file"
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)
    if not any(getattr(handler, "name", "") == "so101_console" for handler in package_logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.name = "so101_console"
        console_handler.setFormatter(formatter)
        package_logger.addHandler(console_handler)
    return log_path


def _is_loopback(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value.lower() == "localhost"


def create_app(
    engine: ControlEngine,
    *,
    control_token: str | None = None,
    agent_service: AgentService | None = None,
    preferences_path: Path | None = None,
) -> Flask:
    """Create the local-only HTTP layer around a single ControlEngine."""

    app = Flask(__name__)
    speech_synthesizer = SpeechSynthesizer()
    app.extensions["so101_speech_synthesizer"] = speech_synthesizer
    token = control_token or secrets.token_urlsafe(32)
    app.config.update(
        MAX_CONTENT_LENGTH=65536,
        SO101_CONTROL_TOKEN=token,
        JSON_SORT_KEYS=False,
    )

    def dispatch_agent_plan(plan: InteractionPlan, context: object) -> dict[str, object]:
        """Translate validated Function Calling actions into existing engine commands.

        The cloud model never receives a joint API.  This adapter is the only bridge
        from the fixed semantic schema to the existing ControlEngine, which remains
        the sole owner of the serial bus and all tuned motion/safety parameters.
        """

        if not isinstance(context, dict):
            raise ControlError("Agent request lost its local control context.")
        operator = str(context.get("client_id", ""))
        if not engine.has_lease(operator):
            raise ControlError("The browser control lease expired before Agent motion.")

        names = [action.name for action in plan.actions]
        follow_actions = {"start_human_follow", "stop_human_follow"} & set(names)
        if follow_actions and len(names) != 1:
            raise ControlError(
                "Start/stop following must be a separate Agent action, not mixed with a gesture."
            )
        if names == ["stop_human_follow"]:
            engine.stop_following("Agent requested stop-following")
            LOGGER.info("agent semantic action dispatched | action=stop_human_follow")
            return {"ok": True, "message": "Following stopped."}
        if names == ["start_human_follow"]:
            LOGGER.info("agent semantic action dispatched | action=start_human_follow")
            return engine.submit(
                "arm",
                {"confirmation": CONTROL_CONFIRMATION_CODE},
                client_id=operator,
            )

        motion_actions: list[dict[str, object]] = []
        source_indices: list[int] = []
        resume = context.get("resume_motion") or {}
        resume_used = False
        def signature(value):
            return {"name": value.get("name"), "arguments": {key: item for key, item in value.get("arguments", {}).items() if key != "speed"}}
        for semantic_index, action in enumerate(plan.actions):
            arguments = dict(action.arguments)
            before = len(motion_actions)
            if (action.name != "return_to_neutral" and not resume_used and resume
                    and signature({"name": action.name, "arguments": arguments}) == signature(resume.get("action", {}))):
                for frame in resume.get("frames", []):
                    if frame["targets"]:
                        args = {"targets": frame["targets"], "hold_s": frame["hold_s"], "return_to_start": False}
                        if frame.get("duration_s") is not None:
                            args["duration_s"] = frame["duration_s"]
                        if frame.get("joint_timing") is not None:
                            args["joint_timing"] = frame["joint_timing"]
                        motion_actions.append({"name": "move_joints", "arguments": args, "_completed_cycles": frame.get("completed_cycles", 0)})
                    else:
                        motion_actions.append({"name": "wait", "arguments": {"seconds": frame["hold_s"]}})
                source_indices.extend([semantic_index] * (len(motion_actions) - before))
                resume_used = True
                continue
            if action.name == "perform_custom_action":
                blocks = action_library.execution_blocks(str(arguments["action_id"]))
                for block in blocks:
                    if block["name"] == "return_to_plan_start":
                        block = {**block, "arguments": {"start_action_index": before}}
                    motion_actions.append(block)
                source_indices.extend([semantic_index] * (len(motion_actions) - before))
                continue
            if action.name in {
                "perform_greeting",
                "open_gripper",
                "close_gripper",
                "enter_pet_pose",
                "leave_pet_pose",
                "return_to_neutral",
                "return_to_action_start",
                "perform_response_gesture",
                "wait",
            }:
                motion_actions.append({"name": action.name, "arguments": arguments})
                source_indices.append(semantic_index)
                continue
            if action.name == "perform_gesture":
                gesture = str(arguments.get("gesture", ""))
                count = int(arguments.get("count", 1))
                intensity = str(arguments.get("intensity", "normal"))
                amplitude = 0.55 if intensity == "small" else 1.0
                if gesture == "nod":
                    motion_actions.append(
                        {
                            "name": "nod",
                            "arguments": {"count": count, "intensity": intensity},
                        }
                    )
                elif gesture == "wave":
                    motion_actions.append(
                        {
                            "name": "wave_wrist",
                            "arguments": {"count": count, "amplitude": amplitude},
                        }
                    )
                else:
                    motion_actions.append(
                        {
                            "name": "happy_wiggle",
                            "arguments": {"count": count, "amplitude": amplitude},
                        }
                    )
                source_indices.extend([semantic_index] * (len(motion_actions) - before))
                continue
            if action.name == "perform_dance":
                duration = float(arguments.get("duration_seconds", 5.0))
                count = max(1, min(3, round(duration / 2.0)))
                motion_actions.extend(
                    [
                        {
                            "name": "happy_wiggle",
                            "arguments": {"count": count, "amplitude": 1.0},
                        },
                        {
                            "name": "wave_wrist",
                            "arguments": {"count": count, "amplitude": 1.0},
                        },
                    ]
                )
                source_indices.extend([semantic_index] * (len(motion_actions) - before))
                continue
            raise ControlError(f"Unsupported local Agent action: {action.name}")

        if len(motion_actions) > MAX_LOCAL_AGENT_ACTIONS:
            raise ControlError("The Agent motion plan is too long; please split the request.")
        if not motion_actions:
            return {"ok": True, "message": "Reply only; no robot action requested."}
        LOGGER.info(
            "agent semantic plan queued | actions=%s",
            [item["name"] for item in motion_actions],
        )
        return engine.submit(
            "agent_plan_start",
            {
                "actions": motion_actions,
                "speech_duration_s": context.get("speech_duration_s"),
                "media_duration_s": context.get("media_duration_s"),
                "playback_rate": context.get("playback_rate", 1.0),
                "semantic_actions": [{"name": action.name, "arguments": dict(action.arguments)} for action in plan.actions],
                "source_indices": source_indices,
            },
            client_id=operator,
        )

    def engine_stop(reason: str) -> dict[str, object]:
        LOGGER.warning("agent stop requested | reason=%s", reason)
        engine.stop_following(reason)
        return {"ok": True, "message": f"DISARMED: {reason}"}

    def agent_stop(reason: str) -> dict[str, object]:
        # Do not let a local Agent dispatch delay the operator's STOP.  The first
        # stop interrupts anything already queued/running; cancel_all then waits
        # out (or invalidates) the dispatch boundary; the final stop covers the
        # narrow case where a dispatch crossed the boundary after the first one.
        speech_synthesizer.cancel_current()
        result = engine_stop(reason)
        try:
            agent_service.cancel_all(reason)
        finally:
            engine_stop(reason)
        return result

    if agent_service is None:
        agent_service = AgentService(
            plan_callback=dispatch_agent_plan,
            stop_callback=engine_stop,
            preferences_path=preferences_path,
        )
    app.extensions["so101_agent_service"] = agent_service

    def client_id() -> str:
        value = request.headers.get("X-Client-Id", "")
        if not CLIENT_ID_PATTERN.fullmatch(value):
            abort(400, "Missing or invalid X-Client-Id header.")
        return value

    def require_operator() -> str:
        value = client_id()
        if not engine.heartbeat(value):
            abort(409, "Another browser tab currently owns the control lease.")
        return value

    def joint_limits():
        if hasattr(engine, "agent_joint_limits"):
            return {"limits": engine.agent_joint_limits(), "calibration_fingerprint": engine._calibration_fingerprint or ""}
        return {"limits": {joint: ((0.0, 100.0) if joint == "gripper.pos" else (-70.0, 70.0)) for joint in (
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")}}

    action_directory = (preferences_path.parent if preferences_path is not None else Path(__file__).resolve().parents[2]) / "action_library"
    action_library = ActionLibrary(action_directory, joint_limits)
    simulation_blueprint, simulation_service = create_simulation_blueprint(
        require_operator=require_operator, limits_provider=joint_limits, library=action_library,
    )
    app.register_blueprint(simulation_blueprint)
    app.extensions["so101_simulation"] = simulation_service
    app.extensions["so101_action_library"] = action_library
    agent_request_gate = threading.RLock()
    client_sequences: dict[str, int] = {}
    reserved_sequences: dict[str, int] = {}
    interruptions: dict[str, dict] = {}

    def accept_sequence(operator: str, payload: dict, *, reserve=False, stop=False):
        if not isinstance(payload, dict):
            abort(400, "请求必须是JSON对象。")
        sequence = payload.get("client_sequence")
        if sequence is None:
            return
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 < sequence < 2**31:
            abort(400, "client_sequence不正确。")
        latest = client_sequences.get(operator, -1)
        reserved = reserved_sequences.get(operator)
        if not stop and (sequence < latest or (sequence == latest and (reserve or reserved != sequence))):
            abort(409, "这条消息已被更新的指令替代。")
        client_sequences[operator] = max(latest, sequence)
        if reserve:
            reserved_sequences[operator] = sequence
        else:
            reserved_sequences.pop(operator, None)
        if len(client_sequences) > 128:
            oldest = next(iter(client_sequences))
            client_sequences.pop(oldest, None)
            reserved_sequences.pop(oldest, None)
            interruptions.pop(oldest, None)

    def preempt_interaction(operator: str) -> dict:
        snapshot = engine.interruption_snapshot() if hasattr(engine, "interruption_snapshot") else {}
        resume = snapshot.pop("_resume", None)
        if resume is None:
            resume = agent_service.interruption_resume(operator)
        context = agent_service.interruption_context(operator, execution_snapshot=snapshot)
        agent_stop(AGENT_INTERACTION_PREEMPT_REASON)
        if snapshot.get("state") in {"RUNNING", "RETURNING", "PAUSED"}:
            engine.submit("agent_interrupt_hold", client_id=operator, timeout_s=3.0)
        return {"context": context, "resume": resume, "created_at": time.monotonic()}

    @app.before_request
    def enforce_local_session() -> None:
        remote = request.remote_addr
        host = urlsplit(f"//{request.host}").hostname
        if not _is_loopback(remote) or not _is_loopback(host):
            abort(403, "This control panel is available only on this computer.")

        protected = request.path.startswith("/api/") or request.path == "/video_feed"
        if protected:
            cookie = request.cookies.get(COOKIE_NAME, "")
            if not secrets.compare_digest(cookie, token):
                abort(403, "Open the control-panel page before using its API.")

        if request.method == "POST":
            expected_origin = request.host_url.rstrip("/")
            if request.headers.get("Origin") != expected_origin:
                abort(403, "Cross-origin control requests are not allowed.")

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(self), geolocation=(), usb=(), serial=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        return response

    @app.errorhandler(ControlError)
    def control_error(exc: ControlError):
        LOGGER.warning("control_error path=%s error=%s", request.path, exc)
        return jsonify(ok=False, error=str(exc)), 409

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            LOGGER.warning(
                "http_error path=%s status=%s error=%s",
                request.path,
                exc.code,
                exc.description,
            )
            return jsonify(ok=False, error=exc.description), exc.code
        return exc

    @app.get("/")
    def index():
        response = make_response(
            render_template(
                "index.html",
                robot_port=engine.config.robot.port,
                robot_id=engine.config.robot.id,
                robot_allowed=engine.robot_allowed,
            )
        )
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="Strict",
            secure=False,
            path="/",
        )
        return response

    @app.get("/api/status")
    def status():
        value = client_id()
        engine.heartbeat(value)
        return jsonify(ok=True, **engine.status(value))

    @app.post("/api/companion")
    def update_companion():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
            abort(400, "陪伴模式开关需要布尔值。")
        allow_unoccupied_idle = payload.get("allow_unoccupied_idle", False)
        allow_vertical_search = payload.get("allow_vertical_search", False)
        expected_epoch = payload.get("expected_epoch")
        if type(allow_unoccupied_idle) is not bool or type(allow_vertical_search) is not bool:
            abort(400, "无人搜索选项需要布尔值。")
        if payload["enabled"] and (type(expected_epoch) is not int or expected_epoch < 0):
            abort(400, "请刷新控制状态后开启陪伴模式。")
        return jsonify(engine.set_companion(
            client_id=operator, enabled=payload["enabled"], allow_unoccupied_idle=allow_unoccupied_idle,
            allow_vertical_search=allow_vertical_search,
            expected_epoch=expected_epoch,
        ))

    @app.post("/api/gaze/calibrate")
    def calibrate_gaze():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if (not isinstance(payload, dict) or not isinstance(payload.get("action"), str)
                or payload["action"] not in {"start", "cancel"}):
            abort(400, "注视标定操作必须是 start 或 cancel。")
        epoch = payload.get("expected_epoch")
        if payload["action"] == "start" and (type(epoch) is not int or epoch < 0):
            abort(400, "请刷新控制状态后启动注视标定。")
        confirmation = payload.get("confirmation", "")
        if not isinstance(confirmation, str):
            abort(400, "注视标定确认字段无效。")
        return jsonify(engine.request_gaze_calibration(
            client_id=operator, action=payload["action"], expected_epoch=epoch,
            confirmation=confirmation,
        ))

    @app.get("/video_feed")
    def video_feed():
        def frames():
            for jpeg in engine.camera_frames():
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/camera/devices")
    def camera_devices():
        client_id()
        try:
            names = list_camera_names()
        except (OSError, RuntimeError) as exc:
            abort(503, str(exc))
        return jsonify(ok=True, devices=[{"name": name} for name in names])

    @app.post("/api/camera/connect")
    def connect_camera():
        operator = require_operator()
        payload = request.get_json(silent=True)
        name = payload.get("name", "") if isinstance(payload, dict) else None
        if not isinstance(name, str) or len(name) > 255 or any(ord(char) < 32 for char in name):
            abort(400, "相机名称格式不正确。")
        return jsonify(engine.connect_camera(name, client_id=operator))

    @app.post("/api/camera/disconnect")
    def disconnect_camera():
        require_operator()
        return jsonify(engine.disconnect_camera())

    @app.post("/api/connect")
    def connect_robot():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        command = {"confirmation": str(payload.get("confirmation", ""))}
        if "port" in payload:
            command["port"] = payload["port"]
        return jsonify(
            engine.submit(
                "connect",
                command,
                client_id=operator,
            )
        )

    @app.get("/api/teach/status")
    def teach_status():
        client_id()
        return jsonify(ok=True, **engine.teach_status())

    @app.post("/api/teach/<path:operation>")
    def teach_operation(operation: str):
        if operation not in {
            "calibration/start", "calibration/middle", "calibration/range",
            "calibration/save", "calibration/cancel", "recording/start", "recording/stop",
        }:
            abort(404)
        operator = require_operator()
        payload = request.get_json(silent=True)
        if payload is not None and not isinstance(payload, dict):
            abort(400, "Expected a JSON object.")
        command = "teach_" + operation.replace("/", "_")
        result = engine.submit(command, payload, client_id=operator)
        recording = result.get("recording")
        if operation == "recording/stop" and recording and recording.get("id"):
            try:
                action = action_library.save_recording(recording)
            except DeletedActionError as exc:
                abort(410, str(exc))
            except (KeyError, ValueError, OSError):
                LOGGER.exception("could not save teach recording | id=%s", recording["id"])
                abort(503, "录制已结束，保存 JSON 失败。请检查项目目录后再次取回录制。")
            result["action"] = {key: action[key] for key in ("id", "name", "description")}
        return jsonify(result)

    @app.post("/api/neutral")
    def collect_neutral():
        operator = require_operator()
        return jsonify(engine.submit("collect_neutral", client_id=operator))

    @app.post("/api/base-calibration/start")
    def start_base_calibration():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        command = {"confirmation": str(payload.get("confirmation", ""))}
        if "port" in payload:
            command["port"] = payload["port"]
        return jsonify(
            engine.submit(
                "start_base_calibration",
                command,
                client_id=operator,
            )
        )

    @app.post("/api/base-calibration/middle")
    def capture_base_middle():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "capture_base_middle",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/base-calibration/range/start")
    def start_base_range():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "start_base_range",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/base-calibration/finish")
    def finish_base_calibration():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "finish_base_calibration",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/base-calibration/cancel")
    def cancel_base_calibration():
        operator = require_operator()
        return jsonify(engine.submit("cancel_base_calibration", client_id=operator))

    @app.post("/api/safe-range/fixed")
    def capture_fixed_positions():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "capture_fixed",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/safe-range/start")
    def start_safe_range():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "start_safe_range",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/safe-range/finish")
    def finish_safe_range():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "finish_safe_range",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/safe-range/cancel")
    def cancel_safe_range():
        operator = require_operator()
        return jsonify(engine.submit("cancel_safe_range", client_id=operator))

    @app.post("/api/arm")
    def arm_robot():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "arm",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/disarm")
    @app.post("/api/stop")
    def disarm_robot():
        operator = require_operator()
        with agent_request_gate:
            accept_sequence(operator, request.get_json(silent=True) or {}, stop=True)
            interruptions.pop(operator, None)
            agent_stop("operator pressed stop-following")
        return jsonify(
            ok=True,
            message="Following stopped. This STOP action did not request torque release.",
        )

    @app.post("/api/disconnect")
    def disconnect_robot():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        agent_service.cancel_all("robot disconnect requested")
        return jsonify(
            engine.submit(
                "disconnect",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    def wait_for_agent_task(task_id: str, timeout_s: float = 35.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            task = agent_service.get_task(task_id)
            if task["state"] not in {"QUEUED", "THINKING", "DISPATCHING"}:
                return task
            time.sleep(0.05)
        raise ControlError("The Agent is still processing; check its status before retrying.")

    def agent_task_response(task: dict[str, object]):
        if task.get("state") == "FAILED":
            return jsonify(ok=False, error=task.get("error") or "Agent request failed."), 502
        response = dict(task)
        response["ok"] = True
        plan = response.get("plan")
        if isinstance(plan, dict):
            response["actions"] = plan.get("actions", [])
            response["requires_confirmation"] = bool(plan.get("requires_confirmation"))
        return jsonify(response)

    @app.get("/api/agent/settings")
    def get_agent_settings():
        value = client_id()
        engine.heartbeat(value)
        return jsonify(ok=True, settings=agent_service.public_settings())

    @app.post("/api/agent/settings")
    def save_agent_settings():
        require_operator()
        payload = request.get_json(silent=True) or {}
        try:
            settings = agent_service.configure(
                agent_name=str(payload.get("agent_name", "")),
                base_url=str(payload.get("base_url", "")),
                model=str(payload.get("model", "")),
                voice=None if "voice" not in payload else str(payload["voice"]),
                language=payload.get("language"),
                api_key=(
                    None
                    if "api_key" not in payload
                    else str(payload.get("api_key", ""))
                ),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            abort(400, str(exc))
        return jsonify(
            ok=True,
            settings=settings,
            message="名称、API 地址、模型、语言和音色已保存。API Key 仅保存在本次后端进程内存中。",
        )

    @app.post("/api/agent/test-connection")
    def test_agent_connection():
        require_operator()
        payload = request.get_json(silent=True) or {}
        try:
            result = agent_service.test_connection(
                agent_name=str(payload.get("agent_name", "")),
                base_url=str(payload.get("base_url", "")),
                model=str(payload.get("model", "")),
                language=payload.get("language"),
                api_key=(
                    None
                    if "api_key" not in payload
                    else str(payload.get("api_key", ""))
                ),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            LOGGER.warning("agent connection test failed | error=%s", exc)
            return jsonify(ok=False, error=str(exc)), 502
        return jsonify(
            ok=True,
            message="连接成功，模型返回了可验证的 Function Calling 结构。",
            reply=result["reply"],
        )

    @app.post("/api/agent/message")
    def send_agent_message():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            abort(400, "消息必须是JSON对象。")
        text = payload.get("text", "")
        if not isinstance(text, str):
            abort(400, "text must be a string.")
        language = payload.get("language")
        if language is not None and language not in ("zh", "en"):
            abort(400, "语言必须选择中文或 English。")
        try:
            with agent_request_gate:
                accept_sequence(operator, payload)
                cached = interruptions.pop(operator, None)
                if cached is not None and time.monotonic() - cached["created_at"] > 15.0:
                    cached = None
                active = engine.status(operator).get("agent_motion", {}).get("active")
                should_interrupt = bool(payload.get("interrupt") or cached or active)
                if should_interrupt:
                    interruption = cached or preempt_interaction(operator)
                else:
                    interruption = {}
                context = {
                    "client_id": operator,
                    "source": str(payload.get("source", "text")),
                    "language": language,
                    "defer_motion": payload.get("defer_motion") is True,
                    "client_sequence": payload.get("client_sequence"),
                    "interrupt": should_interrupt, "interrupt_handled": should_interrupt,
                    "interruption": interruption.get("context"),
                    "resume_motion": interruption.get("resume"),
                    "available_actions": lambda: [
                        {"id": item["id"], "name": item["name"], "description": item.get("description", "")}
                        for item in action_library.list()
                        if item.get("compatible")
                    ],
                }
                queued = agent_service.submit_text(text, context=context)
            task = wait_for_agent_task(str(queued["task_id"]))
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            abort(409, str(exc))
        return agent_task_response(task)

    @app.post("/api/agent/interrupt")
    def interrupt_agent_message():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "插话请求必须是JSON对象。")
        with agent_request_gate:
            accept_sequence(operator, payload, reserve=True)
            interruptions[operator] = preempt_interaction(operator)
        return jsonify(ok=True, message="已暂停旧交互，正在接收补充内容。")

    @app.get("/api/joints")
    def agent_joint_status():
        operator = require_operator()
        return jsonify(engine.agent_joint_status(operator))

    @app.post("/api/joints/move")
    def move_single_agent_joint():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "关节目标必须是JSON对象。")
        joint, position = payload.get("joint"), payload.get("position")
        if not isinstance(joint, str) or joint not in joint_limits()["limits"] or isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position):
            abort(400, "关节名称或目标位置不正确。")
        low, high = joint_limits()["limits"][joint]
        if not low <= position <= high:
            abort(400, "目标超出当前关节安全范围。")
        if not isinstance(payload.get("return_to_start", False), bool):
            abort(400, "回位选项必须是布尔值。")
        with agent_request_gate:
            accept_sequence(operator, payload)
            preempt_interaction(operator)
            interruptions.pop(operator, None)
            return jsonify(engine.submit("agent_plan_start", {"actions": [{"name": "move_joints", "arguments": {
                "targets": {joint: position}, "return_to_start": payload.get("return_to_start", False),
            }}]}, client_id=operator))

    @app.post("/api/actions/<identifier>/execute")
    def execute_saved_action(identifier):
        operator = require_operator()
        try:
            payload = request.get_json(silent=True) if request.is_json else {}
            if not isinstance(payload, dict):
                raise ValueError("动作请求必须是 JSON 对象。")
            actions = action_library.execution_blocks(identifier, playback=payload.get("playback", "keyframes"))
            with agent_request_gate:
                accept_sequence(operator, payload)
                preempt_interaction(operator)
                interruptions.pop(operator, None)
                return jsonify(engine.submit("agent_plan_start", {"actions": actions}, client_id=operator))
        except (ValueError, KeyError) as error:
            abort(400, str(error))

    @app.post("/api/agent/speech")
    def synthesize_agent_speech():
        require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "语音请求必须是JSON对象。")
        task_id = payload.get("task_id")
        if task_id is not None and (not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{32}", task_id)):
            abort(400, "task_id不正确。")
        voice = payload.get("voice", payload.get("preset", "taiwan_female"))
        language = payload.get("language", agent_service.public_settings().get("language", "zh"))
        started = time.monotonic()
        audio = b""
        try:
            audio = speech_synthesizer.synthesize(payload.get("text", ""), voice, language=language)
        except ValueError as exc:
            abort(400, str(exc))
        except SpeechBusyError as exc:
            return jsonify(ok=False, error=str(exc)), 429
        except SpeechTimeoutError as exc:
            return jsonify(ok=False, error=str(exc)), 504
        except SpeechUnavailableError as exc:
            return jsonify(ok=False, error=str(exc)), 503
        except SpeechSynthesisError as exc:
            return jsonify(ok=False, error=str(exc)), 502
        finally:
            if task_id is not None:
                LOGGER.info("agent speech timing | task=%s synthesis_ms=%.0f bytes=%d",
                            task_id, (time.monotonic() - started) * 1000, len(audio))
        response = Response(audio, mimetype="audio/mpeg")
        response.headers["Content-Length"] = str(len(audio))
        return response

    @app.post("/api/agent/latency")
    def report_agent_latency():
        client_id()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "延迟记录必须是JSON对象。")
        task_id = payload.get("task_id")
        source = payload.get("source")
        audio_mode = payload.get("audio_mode")
        interrupted = payload.get("interrupted", False)
        autoplay_blocked = payload.get("autoplay_blocked", False)
        if (not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{32}", task_id)
                or source not in ("text", "voice")
                or audio_mode not in (None, "neural", "system", "system_fallback")
                or type(interrupted) is not bool or type(autoplay_blocked) is not bool):
            abort(400, "延迟记录格式不正确。")
        metrics = {}
        for key in (
            "recognition_ms", "plan_request_ms", "audio_download_ms", "audio_metadata_ms",
            "audio_start_ms", "motion_ack_ms",
        ):
            value = payload.get(key)
            if value is None:
                continue
            if type(value) not in (int, float) or not 0 <= value <= 600_000:
                abort(400, "延迟必须为有限的毫秒数。")
            metrics[key] = round(value)
        LOGGER.info(
            "agent browser latency | task=%s source=%s audio=%s interrupted=%s autoplay_blocked=%s metrics=%s",
            task_id, source, audio_mode, interrupted, autoplay_blocked, metrics,
        )
        return jsonify(ok=True)

    @app.post("/api/agent/start")
    def start_prepared_agent_motion():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "动作启动请求必须是JSON对象。")
        task_id = payload.get("task_id", "")
        if not isinstance(task_id, str) or not task_id or len(task_id) > 96:
            abort(400, "task_id不正确。")
        duration = payload.get("speech_duration_s")
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                abort(400, "speech_duration_s必须是秒数。")
            duration = float(duration)
            if not math.isfinite(duration) or not 0.1 <= duration <= 120.0:
                abort(400, "speech_duration_s必须在0.1到120秒之间。")
        try:
            task = agent_service.start_ready(
                task_id, client_id=operator, speech_duration_s=duration,
                media_duration_s=payload.get("media_duration_s"), playback_rate=payload.get("playback_rate", 1.0),
            )
            task = wait_for_agent_task(str(task["task_id"]), timeout_s=10.0)
        except (RuntimeError, KeyError, ValueError) as exc:
            abort(409, str(exc))
        return agent_task_response(task)

    @app.post("/api/agent/playback")
    def update_agent_playback():
        operator = require_operator()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "播放进度必须是JSON对象。")
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id or len(task_id) > 96:
            abort(400, "task_id不正确。")
        try:
            task = agent_service.get_task(task_id)
        except KeyError:
            return jsonify(ok=True, accepted=False)
        # COMPLETED means dispatched, not physically finished.  Bind updates to
        # the motion epoch returned by that dispatch, not the service state.
        motion = (task.get("result") or {}).get("agent_motion") or {}
        motion_id = motion.get("motion_id")
        if isinstance(motion_id, bool) or not isinstance(motion_id, int):
            return jsonify(ok=True, accepted=False)
        try:
            accepted = engine.update_agent_playback(
                client_id=operator, motion_id=motion_id,
                position_s=payload.get("position_s"), duration_s=payload.get("duration_s"),
                state=payload.get("state"), sequence=payload.get("sequence"),
                playback_rate=payload.get("playback_rate", 1.0),
            )
        except ValueError as exc:
            abort(400, str(exc))
        except ControlError as exc:
            abort(409, str(exc))
        return jsonify(ok=True, accepted=accepted)

    @app.post("/api/agent/confirm")
    def confirm_agent_message():
        require_operator()
        payload = request.get_json(silent=True) or {}
        task_id = str(payload.get("task_id", ""))
        try:
            task = agent_service.confirm(task_id)
            task = wait_for_agent_task(str(task["task_id"]), timeout_s=10.0)
        except (RuntimeError, KeyError, ValueError) as exc:
            abort(409, str(exc))
        return agent_task_response(task)

    @app.post("/api/agent/cancel")
    def cancel_agent_message():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        task_id = str(payload.get("task_id", ""))
        discard_only = payload.get("discard_only", False)
        if not isinstance(discard_only, bool):
            abort(400, "discard_only必须为布尔值。")
        if discard_only and not task_id:
            abort(400, "丢弃待执行任务时必须指定task_id。")
        cancelled_task = None
        if not task_id:
            active = next(
                (
                    item
                    for item in agent_service.history()
                    if item.get("state") not in {"COMPLETED", "FAILED", "CANCELLED"}
                ),
                None,
            )
            task_id = "" if active is None else str(active["task_id"])
        if task_id:
            try:
                cancelled_task = agent_service.cancel(task_id)
            except KeyError:
                cancelled_task = None
        if discard_only:
            # A rejected new task must never stop the unrelated, older motion
            # which caused its rejection.  Explicit operator cancellation below
            # retains the normal stop behaviour.
            LOGGER.info("agent pending task discarded | task=%s stop_motion=false", task_id)
            return jsonify(ok=True, message="本条待执行任务已丢弃。", task=cancelled_task)
        with agent_request_gate:
            accept_sequence(operator, payload, stop=True)
            interruptions.pop(operator, None)
            agent_stop("operator cancelled Agent interaction")
        return jsonify(
            ok=True,
            message="当前 Agent 交互和机械臂动作已取消。",
            task=cancelled_task,
        )

    @app.get("/api/agent/status")
    def agent_status():
        value = client_id()
        engine.heartbeat(value)
        settings = agent_service.public_settings()
        history = agent_service.history()
        latest = history[0] if history else None
        robot_status = engine.status(value)
        motion = robot_status.get("agent_motion", {})
        state = "idle"
        message = "等待你的指令"
        if isinstance(motion, dict) and motion.get("active"):
            state = "executing"
            message = str(motion.get("message") or "正在执行动作")
        elif latest is not None:
            state_map = {
                "QUEUED": "thinking",
                "THINKING": "thinking",
                "DISPATCHING": "executing",
                "READY": "idle",
                "WAITING_CONFIRMATION": "idle",
                "FAILED": "error",
            }
            state = state_map.get(str(latest.get("state")), "idle")
            message = str(latest.get("error") or latest.get("reply") or message)
        return jsonify(
            ok=True,
            status={
                **settings,
                "state": state,
                "message": message,
                "latest_task": latest,
                "motion": motion,
                "visual_proximity": robot_status.get("visual_proximity"),
            },
        )

    @app.get("/api/agent/history")
    def agent_history():
        value = client_id()
        engine.heartbeat(value)
        messages: list[dict[str, str]] = []
        for item in reversed(agent_service.history()):
            original = str(item.get("original_text") or "")
            if original:
                messages.append({"role": "user", "content": original})
            reply = str(item.get("reply") or "")
            error = str(item.get("error") or "")
            if reply or error:
                plan = item.get("plan")
                action_summary = ""
                if isinstance(plan, dict):
                    action_summary = "动作：" + "、".join(
                        str(action.get("name", ""))
                        for action in plan.get("actions", [])
                        if isinstance(action, dict) and action.get("name")
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply or f"执行失败：{error}",
                        "action_summary": action_summary,
                    }
                )
        return jsonify(ok=True, history=messages)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local browser control panel for SO-101 human pose following."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="Allow the page to connect to the configured robot. Default is preview-only.",
    )
    parser.add_argument("--camera-index", type=int, help="Override camera index")
    parser.add_argument("--robot-port", help="Override serial port, for example COM4")
    parser.add_argument("--robot-id", help="Override the saved LeRobot calibration id")
    parser.add_argument("--http-port", type=int, default=7860, help="Local HTTP port")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.http_port <= 65535:
        raise SystemExit("--http-port must be between 1 and 65535.")

    config_path = Path(args.config).expanduser().resolve()
    log_path = configure_backend_logging(config_path.parent)
    LOGGER.info("backend_start config=%s robot_enabled=%s", config_path, args.enable_robot)
    config = load_config(config_path)
    if args.camera_index is not None:
        config = replace(config, camera=replace(config.camera, index=args.camera_index))
    if args.robot_port or args.robot_id:
        config = replace(
            config,
            robot=replace(
                config.robot,
                port=args.robot_port or config.robot.port,
                id=args.robot_id or config.robot.id,
            ),
        )

    profile_path = config_path.parent / "safety_profile.json"
    engine = ControlEngine(
        config,
        robot_allowed=args.enable_robot,
        profile_path=profile_path,
    )
    app = create_app(engine, preferences_path=config_path.parent / "agent_preferences.json")
    server = make_server("127.0.0.1", args.http_port, app, threaded=True)
    address = f"http://127.0.0.1:{args.http_port}/"

    print(f"SO-101 local control panel: {address}")
    print(f"Backend log: {log_path}")
    print("Preview mode" if not args.enable_robot else "Robot connection ENABLED")
    print(
        "Web STOP requests DISARM but cannot cancel a confirmed torque release; "
        "the physical power switch is the real emergency stop."
    )
    engine.start()
    if not args.no_browser:
        webbrowser.open(address)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local control panel...")
    finally:
        app.extensions["so101_simulation"].shutdown()
        app.extensions["so101_speech_synthesizer"].cancel_current()
        app.extensions["so101_agent_service"].shutdown()
        engine.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
