from __future__ import annotations

import argparse
import ipaddress
import logging
import re
import secrets
import sys
import webbrowser
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, make_response, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from .config import load_config
from .web_engine import ControlEngine, ControlError


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
) -> Flask:
    """Create the local-only HTTP layer around a single ControlEngine."""

    app = Flask(__name__)
    token = control_token or secrets.token_urlsafe(32)
    app.config.update(
        MAX_CONTENT_LENGTH=4096,
        SO101_CONTROL_TOKEN=token,
        JSON_SORT_KEYS=False,
    )

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
            "camera=(), microphone=(), geolocation=(), usb=(), serial=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; object-src 'none'; "
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

    @app.get("/video_feed")
    def video_feed():
        def frames():
            revision = -1
            while True:
                next_revision, jpeg = engine.wait_for_frame(revision, timeout_s=2.0)
                if jpeg is None or next_revision == revision:
                    if engine.status(None).get("preview_state") == "STOPPED":
                        break
                    continue
                revision = next_revision
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

    @app.post("/api/connect")
    def connect_robot():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "connect",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

    @app.post("/api/neutral")
    def collect_neutral():
        operator = require_operator()
        return jsonify(engine.submit("collect_neutral", client_id=operator))

    @app.post("/api/base-calibration/start")
    def start_base_calibration():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "start_base_calibration",
                {"confirmation": str(payload.get("confirmation", ""))},
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
        require_operator()
        engine.stop_following("operator pressed stop-following")
        return jsonify(
            ok=True,
            message="Following stopped. This STOP action did not request torque release.",
        )

    @app.post("/api/disconnect")
    def disconnect_robot():
        operator = require_operator()
        payload = request.get_json(silent=True) or {}
        return jsonify(
            engine.submit(
                "disconnect",
                {"confirmation": str(payload.get("confirmation", ""))},
                client_id=operator,
            )
        )

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
    app = create_app(engine)
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
        engine.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
