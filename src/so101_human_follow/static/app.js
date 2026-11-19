(() => {
  "use strict";

  const CLIENT_KEY = "so101-human-follow-client-id";
  const DEFAULT_WRIST_CAMERA = "icspring camera";
  const COMPANION_CAMERA_TIMEOUT_MS = 12_000;
  const POLL_INTERVAL_MS = 500;
  const ACTION_COUNTDOWN_SECONDS = 3;
  const ARM_ACQUIRE_TIMEOUT_SECONDS = 30;
  const CONTROL_CONFIRMATION_CODE = "123";
  const clientId = getClientId();
  let currentState = null;
  let pollingPromise = null;
  let busy = false;
  let pendingCommand = null;
  let videoFailed = false;
  let videoSession = null;
  let cameraCommand = null;
  let cameraCommandError = "";
  let disconnectCameraAfterConnect = false;
  let cameraRetainedForCompanion = false;
  let companionStart = null;
  let companionWasEnabled = false;
  let companionCommandError = "";
  let gazeCommand = null;
  let gazeCommandError = "";
  let gazeRequestSequence = 0;
  let neutralCountdownTimer = null;
  let neutralCountdownRemaining = 0;
  let neutralCommandError = "";
  let armCountdownTimer = null;
  let armCountdownRemaining = 0;
  let armAcquireRemaining = 0;
  let armCountdownPhase = "idle";
  let followerPortEdited = false;

  const elements = {
    globalMessage: byId("globalMessage"),
    operatorBadge: byId("operatorBadge"),
    armBadge: byId("armBadge"),
    previewBadge: byId("previewBadge"),
    robotBadge: byId("robotBadge"),
    readOnlyNotice: byId("readOnlyNotice"),
    videoDialog: byId("videoFollowDialog"),
    cameraPanel: byId("cameraPanel"),
    cameraPanelTitle: byId("cameraPanelTitle"),
    videoCameraSlot: byId("videoCameraSlot"),
    companionCameraSlot: byId("companionCameraSlot"),
    videoFeed: byId("videoFeed"),
    videoFallback: byId("videoFallback"),
    videoFallbackTitle: byId("videoFallbackTitle"),
    cameraForm: byId("cameraForm"),
    cameraName: byId("cameraName"),
    cameraDevices: byId("cameraDevices"),
    cameraConnectButton: byId("cameraConnectButton"),
    cameraDisconnectButton: byId("cameraDisconnectButton"),
    cameraStatus: byId("cameraStatus"),
    companionDialog: byId("companionDialog"),
    companionWorkspaceDialog: byId("companionWorkspaceDialog"),
    companionEnterButton: byId("companionEnterButton"),
    companionDialogStopButton: byId("companionDialogStopButton"),
    companionRobotForm: byId("companionRobotForm"),
    companionFollowerPort: byId("companionFollowerPort"),
    companionConnectButton: byId("companionConnectButton"),
    companionDisconnectButton: byId("companionDisconnectButton"),
    companionRobotStatus: byId("companionRobotStatus"),
    companionBadge: byId("companionBadge"),
    companionStatus: byId("companionStatus"),
    companionUnoccupied: byId("companionUnoccupied"),
    companionToggleButton: byId("companionToggleButton"),
    gazeCalibrationBadge: byId("gazeCalibrationBadge"),
    gazeCalibrationStartButton: byId("gazeCalibrationStartButton"),
    gazeCalibrationCancelButton: byId("gazeCalibrationCancelButton"),
    gazeCalibrationProgress: byId("gazeCalibrationProgress"),
    gazeCalibrationStatus: byId("gazeCalibrationStatus"),
    gazeCalibrationLastErrorDetails: byId("gazeCalibrationLastErrorDetails"),
    gazeCalibrationLastErrorSummary: byId("gazeCalibrationLastErrorSummary"),
    gazeCalibrationLastError: byId("gazeCalibrationLastError"),
    gazeCenterError: byId("gazeCenterError"),
    poseDot: byId("poseDot"),
    poseSummary: byId("poseSummary"),
    frameAge: byId("frameAge"),
    poseReason: byId("poseReason"),
    peopleCount: byId("peopleCount"),
    validStreak: byId("validStreak"),
    sendAge: byId("sendAge"),
    robotIdentity: byId("robotIdentity"),
    followerPort: byId("followerPort"),
    safeRangeStep: byId("safeRangeStep"),
    safeRangeTestNotice: byId("safeRangeTestNotice"),
    neutralStepIndex: byId("neutralStepIndex"),
    armStepIndex: byId("armStepIndex"),
    baseCalibrationProgressBar: byId("baseCalibrationProgressBar"),
    baseCalibrationProgressText: byId("baseCalibrationProgressText"),
    baseCalibrationValues: byId("baseCalibrationValues"),
    rangeProgressBar: byId("rangeProgressBar"),
    rangeProgressText: byId("rangeProgressText"),
    neutralProgressBar: byId("neutralProgressBar"),
    neutralProgressText: byId("neutralProgressText"),
    leaseState: byId("leaseState"),
    robotState: byId("robotState"),
    previewState: byId("previewState"),
    cameraConfig: byId("cameraConfig"),
    poseSide: byId("poseSide"),
    jointRows: byId("jointRows"),
    connectionState: byId("connectionState"),
    connectButton: byId("connectButton"),
    baseCalibrationStartButton: byId("baseCalibrationStartButton"),
    baseCalibrationMiddleButton: byId("baseCalibrationMiddleButton"),
    baseCalibrationRangeStartButton: byId("baseCalibrationRangeStartButton"),
    baseCalibrationFinishButton: byId("baseCalibrationFinishButton"),
    baseCalibrationCancelButton: byId("baseCalibrationCancelButton"),
    fixedButton: byId("fixedButton"),
    rangeStartButton: byId("rangeStartButton"),
    rangeFinishButton: byId("rangeFinishButton"),
    rangeCancelButton: byId("rangeCancelButton"),
    neutralButton: byId("neutralButton"),
    armButton: byId("armButton"),
    stopButton: byId("stopButton"),
    disconnectButton: byId("disconnectButton"),
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function getClientId() {
    // Generate for every page instance. Some browsers clone sessionStorage when a
    // tab is duplicated; reusing a stored id would let both tabs share one lease.
    const id = globalThis.crypto?.randomUUID?.() ||
      `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    // Robot controls and Agent requests must share this document's identity,
    // even if browser storage is cleared or unavailable while the page is open.
    Object.defineProperty(window, "so101ControlClientId", { value: id });
    try { sessionStorage.setItem(CLIENT_KEY, id); } catch (_error) { /* optional only */ }
    return id;
  }

  function formatAge(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return "—";
    }
    const seconds = Number(value);
    if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
    return `${seconds.toFixed(1)} s`;
  }

  function formatValue(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return "—";
    }
    return Number(value).toFixed(2);
  }

  function setBadge(element, text, kind = "muted") {
    element.textContent = text;
    element.className = `badge badge-${kind}`;
  }

  function badgeKind(state) {
    const value = String(state || "").toUpperCase();
    if (value.includes("FAULT") || value.includes("ERROR")) return "danger";
    if (value === "ARMED") return "armed";
    if (value === "CONNECTED_DISARMED" || value === "RUNNING") return "safe";
    if (value.includes("ING") || value === "STARTING") return "warning";
    return "muted";
  }

  function translatePoseReason(reason) {
    const labels = {
      ok: "姿态正常",
      "no person": "未检测到人",
      "multiple people": "检测到多人",
      "arm landmarks unreliable": "手臂关键点不可靠",
      "no hand": "未检测到目标手",
      "hand landmarks unreliable": "手部关键点不可靠",
      "selected hand not found": "未找到对应的右手",
      "pose or hand landmarks unreliable": "人体或手部关键点不可靠",
      "no reliable control channel": "当前没有可靠控制通道",
      "pose + hand channels active": "手臂与手部通道有效",
      "pose channels active; hand held": "手臂通道有效；手部舵机保持",
      "hand channels active; arm held": "手部通道有效；手臂舵机保持",
      "person changed": "人物发生切换",
      "not started": "尚未启动",
    };
    return labels[reason] || reason || "—";
  }

  function neutralCountdownAllowed(state) {
    return Boolean(
      state?.operator &&
      state.robot_state === "CONNECTED_DISARMED" &&
      state.preview_state === "RUNNING" &&
      cameraCommand !== "disconnect" &&
      !state.armed &&
      !state.agent_motion?.active &&
      !state.neutral_collecting &&
      !state.release_pending &&
      !gazeCalibrationActive(state) &&
      (!state.robot_allowed || state.safe_range?.profile_ready) &&
      !state.error
    );
  }

  function armCountdownSafe(state) {
    return Boolean(
      state?.operator &&
      state.robot_state === "CONNECTED_DISARMED" &&
      state.preview_state === "RUNNING" &&
      cameraCommand !== "disconnect" &&
      state.neutral_ready &&
      !state.armed &&
      !state.agent_motion?.active &&
      !state.neutral_collecting &&
      !state.release_pending &&
      !gazeCalibrationActive(state) &&
      !state.error
    );
  }

  function render(state) {
    currentState = state;
    const robotState = state.robot_state || "UNKNOWN";
    const previewState = state.preview_state || "UNKNOWN";
    const operator = Boolean(state.operator);
    const poseValid = Boolean(state.pose_valid);
    const safeRange = state.safe_range || {};
    const agentMotionActive = Boolean(state.agent_motion?.active);
    const gazeCalibrating = gazeCalibrationActive(state);
    const safeRangeSkipped = Boolean(
      safeRange.skipped || safeRange.test_mode || state.skip_safe_range_calibration,
    );
    const baseCalibration = state.base_calibration || {};
    const baseCalibrationActive = Boolean(baseCalibration.active);
    const rangeRecording = safeRange.state === "RECORDING";
    const fixedPositioning = safeRange.state === "POSITIONING_FIXED";
    const rangeWaitingReturn = safeRange.state === "WAITING_RETURN";
    const canScheduleNeutral = neutralCountdownAllowed(state);
    const armStateSafe = armCountdownSafe(state);
    // Safe-range capture is intentionally skipped in the current test flow.
    // Keep the legacy controls in the DOM for API compatibility, but never
    // expose this obsolete step to the operator.
    elements.safeRangeStep.hidden = true;
    elements.safeRangeTestNotice.hidden = !safeRangeSkipped;
    elements.neutralStepIndex.textContent = "1";
    elements.armStepIndex.textContent = "2";
    let countdownCancelled = false;
    if (neutralCountdownTimer !== null && !canScheduleNeutral) {
      cancelNeutralCountdown(false);
      neutralCommandError = "连接或控制权已变化，请确认后重新采集。";
      countdownCancelled = true;
    }
    // A person may need to leave the arm-mounted camera view in order to click
    // this page.  Keep the countdown alive while the static safety prerequisites
    // remain valid; live pose validity is checked once, immediately before ARM.
    if (armCountdownTimer !== null && !armStateSafe) {
      cancelArmCountdown(false);
      countdownCancelled = true;
    }
    const countdownActive =
      neutralCountdownTimer !== null || armCountdownTimer !== null;

    setBadge(
      elements.operatorBadge,
      operator ? "本页面持有控制权" : "只读 · 其他页面控制",
      operator ? "safe" : "warning",
    );
    setBadge(
      elements.armBadge,
      state.armed ? "ARMED · 正在跟随" : "DISARMED · 未跟随",
      state.armed ? "armed" : "safe",
    );
    const previewLabels = {
      STOPPED: "未连接", STARTING: "连接中", RUNNING: "已连接", STOPPING: "断开中", FAULT: "相机故障",
    };
    setBadge(elements.previewBadge, previewLabels[previewState] || previewState, badgeKind(previewState));
    setBadge(elements.robotBadge, robotState, badgeKind(robotState));

    elements.readOnlyNotice.hidden = operator;
    elements.globalMessage.textContent = state.error || (
      countdownCancelled
        ? "控制权、连接或识别状态发生变化，倒计时已自动取消。"
        : state.message || "后台运行中"
    );
    elements.globalMessage.classList.toggle(
      "is-error",
      Boolean(state.error || countdownCancelled),
    );
    elements.globalMessage.classList.toggle("is-armed", Boolean(state.armed));

    const headVisible = Boolean(state.companion?.head_visible);
    elements.poseDot.classList.toggle("is-good", poseValid || headVisible);
    elements.poseDot.classList.toggle("is-bad", !poseValid && !headVisible);
    const channelLabels = {
      lift: "手臂抬高",
      elbow: "肘部弯曲",
      hand_roll: "手掌旋转",
      hand_open: "张手/握拳",
    };
    const validChannels = Array.isArray(state.valid_channels) ? state.valid_channels : [];
    const gazeSummary = {
      uncalibrated: "等待注视标定", calibrating: "正在标定方向", waiting: "标定等待准备", preparing: "标定等待准备",
      tracking: "正在纠偏", centered: "已居中", fault: "纠偏已暂停",
      pose_changed: "需要重新标定", camera_changed: "需要重新标定",
    }[state.companion?.gaze?.state];
    elements.poseSummary.textContent = headVisible ? `头部已定位${gazeSummary ? ` · ${gazeSummary}` : ""}` : validChannels.length
      ? `有效：${validChannels.map((name) => channelLabels[name] || name).join("、")}`
      : "暂无有效控制通道";
    elements.frameAge.textContent = `帧延迟 ${formatAge(state.frame_age_s)}`;
    elements.poseReason.textContent = headVisible ? "已定位头部" : translatePoseReason(state.pose_reason);
    elements.peopleCount.textContent = !state.people_count && headVisible ? "1（头部）" : state.people_count ?? "—";
    elements.validStreak.textContent = state.valid_streak ?? "—";
    elements.sendAge.textContent = formatAge(state.last_send_age_s);

    const followerPort = state.robot_port || state.port || "";
    if (state.robot_connected || (!followerPortEdited && followerPort)) {
      elements.followerPort.value = followerPort;
    }
    elements.robotIdentity.textContent = `${state.robot_id || "SO-101"} · ${followerPort || "—"}`;
    elements.leaseState.textContent = operator
      ? `本页面 · 心跳 ${formatAge(state.lease_age_s)}`
      : `只读 · ${formatAge(state.lease_age_s)}`;
    elements.robotState.textContent = robotState;
    elements.previewState.textContent = previewState;
    const camera = state.camera || {};
    elements.cameraConfig.textContent =
      `${state.camera_name || `#${camera.index ?? "—"}`} · ${camera.width ?? "—"}×${camera.height ?? "—"} · ${camera.fps ?? "—"} FPS · 校正 ${camera.rotation_degrees ?? 0}°`;
    elements.poseSide.textContent =
      `${state.pose_side === "left" ? "左臂 + 左手" : "右臂 + 右手"} · ${state.control_hz ?? "—"} Hz`;

    const baseTimeout = Math.max(1, Number(baseCalibration.timeout_s) || 90);
    const baseRemaining = Math.max(0, Number(baseCalibration.remaining_s) || 0);
    const baseStagePercent = {
      IDLE: 0,
      STARTING: 5,
      WAITING_MIDDLE: 15,
      MIDDLE_CAPTURED: 25,
      RANGE_INCOMPLETE: 90,
      SAVING: 95,
      COMPLETE: 100,
      FAULT: 0,
    };
    const basePercent = baseCalibration.state === "RECORDING"
      ? Math.min(90, 25 + Math.round(((baseTimeout - baseRemaining) / baseTimeout) * 65))
      : (baseStagePercent[baseCalibration.state] ?? 0);
    elements.baseCalibrationProgressBar.value = basePercent;
    elements.baseCalibrationProgressBar.setAttribute("aria-valuenow", String(basePercent));
    const baseMessages = {
      IDLE: "连接失败并提示 calibration mismatch 时，从这里开始",
      STARTING: `正在打开 ${followerPort || "从臂串口"} 并释放全部六轴扭矩…`,
      WAITING_MIDDLE: "六轴已释放：请托住机械臂，摆到各自行程中间，再点击“② 确定当前六轴中位”",
      MIDDLE_CAPTURED: "中位已确定：请托住机械臂，再点击“③ 开始 90 秒范围采集”",
      RECORDING: `正在记录六轴原始范围 · 剩余 ${Math.ceil(baseRemaining)} 秒；请依次移动 ID1–ID6`,
      RANGE_INCOMPLETE: `范围不完整：${baseCalibration.error || "请补充移动提示的舵机后重试"}`,
      SAVING: "正在写入舵机和校准 JSON…",
      COMPLETE: "基础校准已保存，现在可以点击“连接机器人”",
      FAULT: `基础校准失败：${baseCalibration.error || "未知错误"}`,
    };
    elements.baseCalibrationProgressText.textContent =
      baseMessages[baseCalibration.state] || baseCalibration.state || "未开始";
    const rawCurrent = baseCalibration.current_raw || {};
    const rawMins = baseCalibration.mins_raw || {};
    const rawMaxes = baseCalibration.maxes_raw || {};
    const motorLabels = {
      shoulder_pan: "ID1",
      shoulder_lift: "ID2",
      elbow_flex: "ID3",
      wrist_flex: "ID4",
      wrist_roll: "ID5",
      gripper: "ID6",
    };
    elements.baseCalibrationValues.textContent = Object.keys(motorLabels)
      .filter((name) => rawCurrent[name] !== undefined)
      .map((name) => `${motorLabels[name]} ${rawMins[name] ?? "—"}/${rawCurrent[name]}/${rawMaxes[name] ?? "—"}`)
      .join(" · ");

    const rangeTimeout = Math.max(
      1,
      Number(safeRange.timeout_s) || (fixedPositioning ? 10 : 60),
    );
    const rangeRemaining = Math.max(0, Number(safeRange.remaining_s) || 0);
    let rangePercent = 0;
    if (safeRange.profile_ready) {
      rangePercent = 100;
    } else if (fixedPositioning) {
      rangePercent = Math.min(
        20,
        Math.round(((rangeTimeout - rangeRemaining) / rangeTimeout) * 20),
      );
    } else if (safeRange.state === "STARTING") {
      rangePercent = 20;
    } else if (rangeRecording) {
      rangePercent = Math.min(
        85,
        20 + Math.round(((rangeTimeout - rangeRemaining) / rangeTimeout) * 65),
      );
    } else if (rangeWaitingReturn) {
      rangePercent = 90;
    } else if (safeRange.state === "FINISHING") {
      rangePercent = 95;
    }
    elements.rangeProgressBar.value = rangePercent;
    elements.rangeProgressBar.setAttribute("aria-valuenow", String(rangePercent));
    if (safeRange.profile_ready) {
      elements.rangeProgressText.textContent = "已完成 · ID1–ID2 固定，ID3–ID6 安全范围已保存";
    } else if (fixedPositioning) {
      elements.rangeProgressText.textContent =
        `正在设置 ID1–ID2 固定位置 · 剩余 ${Math.ceil(rangeRemaining)} 秒；请托稳并摆到目标位置`;
    } else if (safeRange.state === "STARTING") {
      elements.rangeProgressText.textContent = "ID1–ID2 正在锁定，随后自动释放 ID3–ID6…";
    } else if (rangeRecording) {
      elements.rangeProgressText.textContent =
        `正在自动采集 ID3–ID6 · 剩余 ${Math.ceil(rangeRemaining)} 秒；请逐轴摆到允许边界`;
    } else if (rangeWaitingReturn) {
      elements.rangeProgressText.textContent = safeRange.profile_error
        ? `60 秒数据已保留 · 上次保存未通过：${safeRange.profile_error} · 调整后再次点击保存，无需重做 ID1–ID2 或 60 秒采集`
        : "60 秒采集完成 · 请把 ID3–ID6 放回舒适中位，再点击“已回到中位·保存并锁定”";
    } else if (safeRange.fixed_ready) {
      elements.rangeProgressText.textContent = safeRange.profile_error
        ? `固定位置已确认 · 上次范围无效：${safeRange.profile_error}`
        : "ID1–ID2 已锁定 · 等待自动开始 ID3–ID6 范围采集";
    } else {
      elements.rangeProgressText.textContent = safeRange.profile_error
        ? `未完成：${safeRange.profile_error}`
        : "未完成 · 先设置 ID1–ID2 固定位置；数值不是角度";
    }

    const neutralTimeout = Math.max(1, Number(state.neutral_timeout_s) || 30);
    const neutralRemaining = Math.max(0, Number(state.neutral_remaining_s) || 0);
    const percent = state.neutral_ready
      ? 100
      : state.neutral_collecting
        ? Math.min(
          100,
          Math.round(((neutralTimeout - neutralRemaining) / neutralTimeout) * 100),
        )
        : 0;
    elements.neutralProgressBar.value = percent;
    elements.neutralProgressBar.setAttribute("aria-valuenow", String(percent));
    if (neutralCountdownTimer !== null) {
      elements.neutralProgressText.textContent = `准备入镜 · ${neutralCountdownRemaining} 秒后开始采集`;
    } else if (pendingCommand === "/api/neutral" && !state.neutral_collecting) {
      elements.neutralProgressText.textContent = "正在启动 30 秒采集…";
    } else if (neutralCommandError) {
      elements.neutralProgressText.textContent = `采集未启动：${neutralCommandError}`;
    } else if (state.neutral_ready) {
      elements.neutralProgressText.textContent = "已完成 · 可以 ARM";
    } else if (state.neutral_collecting) {
      const perChannel = state.neutral_channel_progress || {};
      const detail = Object.entries(channelLabels)
        .map(([name, label]) => `${label} ${Number(perChannel[name]) || 0} 帧`)
        .join(" · ");
      const status = poseValid ? "正在采集" : "已开始 · 等待右肩、右肘和右手入镜";
      elements.neutralProgressText.textContent = `${status} · 剩余 ${Math.ceil(neutralRemaining)} 秒 · ${detail}`;
    } else {
      elements.neutralProgressText.textContent =
        state.neutral_error ? `采集未完成：${state.neutral_error}` : "未完成 · 将连续采集 30 秒";
    }

    elements.connectButton.disabled = busy || countdownActive || !operator || !state.can_connect;
    elements.followerPort.disabled = busy || !operator || Boolean(state.robot_connected) ||
      baseCalibrationActive || robotState === "CONNECTING" || Boolean(state.teach_active);
    elements.baseCalibrationStartButton.disabled =
      busy || countdownActive || !operator || !state.can_start_base_calibration;
    elements.baseCalibrationMiddleButton.disabled =
      busy || countdownActive || !operator || !state.can_capture_base_middle;
    elements.baseCalibrationRangeStartButton.disabled =
      busy || countdownActive || !operator || !state.can_start_base_range;
    elements.baseCalibrationFinishButton.disabled =
      busy || countdownActive || !operator || !state.can_finish_base_calibration;
    elements.baseCalibrationCancelButton.disabled =
      busy || countdownActive || !operator || !state.can_cancel_base_calibration;
    elements.fixedButton.disabled =
      safeRangeSkipped || busy || countdownActive || !operator || !state.can_capture_fixed;
    // ID3–ID6 range capture starts automatically after the 10-second ID1–ID2
    // positioning phase; this button is a phase indicator, not a second command.
    elements.rangeStartButton.disabled = true;
    elements.rangeFinishButton.disabled =
      safeRangeSkipped || busy || countdownActive || !operator || !state.can_finish_safe_range;
    elements.rangeCancelButton.disabled =
      safeRangeSkipped || busy || countdownActive || !operator || !state.can_cancel_safe_range;
    [
      elements.baseCalibrationStartButton,
      elements.baseCalibrationMiddleButton,
      elements.baseCalibrationRangeStartButton,
      elements.baseCalibrationFinishButton,
      elements.fixedButton,
      elements.rangeFinishButton,
      elements.neutralButton,
      elements.armButton,
    ].forEach((button) => button.classList.remove("is-next-step"));
    if (!elements.baseCalibrationStartButton.disabled) {
      elements.baseCalibrationStartButton.classList.add("is-next-step");
    } else if (!elements.baseCalibrationMiddleButton.disabled) {
      elements.baseCalibrationMiddleButton.classList.add("is-next-step");
    } else if (!elements.baseCalibrationRangeStartButton.disabled) {
      elements.baseCalibrationRangeStartButton.classList.add("is-next-step");
    } else if (!elements.baseCalibrationFinishButton.disabled && baseCalibration.state === "RANGE_INCOMPLETE") {
      elements.baseCalibrationFinishButton.classList.add("is-next-step");
    }
    elements.neutralButton.disabled = neutralCountdownTimer !== null
      ? false
      : busy || armCountdownTimer !== null || !canScheduleNeutral;
    elements.armButton.disabled = armCountdownTimer !== null
      ? false
      : busy || neutralCountdownTimer !== null || !armStateSafe;
    if (!baseCalibrationActive && !state.robot_connected && !elements.connectButton.disabled) {
      elements.connectButton.classList.add("is-next-step");
    } else {
      elements.connectButton.classList.remove("is-next-step");
    }
    if (!safeRangeSkipped && !elements.fixedButton.disabled && !safeRange.profile_ready) {
      elements.fixedButton.classList.add("is-next-step");
    }
    if (!safeRangeSkipped && !elements.rangeFinishButton.disabled && rangeWaitingReturn) {
      elements.rangeFinishButton.classList.add("is-next-step");
    }
    if (!elements.neutralButton.disabled && safeRange.profile_ready && !state.neutral_ready) {
      elements.neutralButton.classList.add("is-next-step");
    }
    if (!elements.armButton.disabled && state.neutral_ready && !state.armed) {
      elements.armButton.classList.add("is-next-step");
    }
    const releaseInProgress = Boolean(
      state.release_pending ||
      robotState === "DISCONNECTING" ||
      pendingCommand === "/api/disconnect"
    );
    const armPending = pendingCommand === "/api/arm";
    const neutralPending = pendingCommand === "/api/neutral" || state.neutral_collecting;
    // STOP can cancel a local countdown, but a confirmed torque release cannot be cancelled.
    elements.stopButton.disabled =
      !operator || releaseInProgress || (!countdownActive && !state.armed && !armPending && !neutralPending && !agentMotionActive &&
        !state.teach_following && !state.companion?.enabled && !companionStart && !gazeCalibrating);
    elements.disconnectButton.disabled =
      busy || countdownActive || !operator || !state.can_disconnect;
    elements.connectButton.textContent = robotState === "CONNECTING"
      ? "连接中…"
      : state.robot_connected
        ? robotState === "FAULT" ? "已连接 · 需处理故障" : "已连接从臂"
        : "连接机器人";
    elements.fixedButton.textContent = fixedPositioning
      ? `正在设置 ID1–ID2 · ${Math.ceil(Number(safeRange.remaining_s) || 0)} 秒`
      : safeRange.fixed_ready
        ? "重新设置 ID1–ID2 固定位置（10 秒）"
        : "设置 ID1–ID2 固定位置（10 秒）";
    elements.rangeStartButton.textContent = rangeRecording
      ? `正在自动采集 ID3–ID6 · ${Math.ceil(Number(safeRange.remaining_s) || 0)} 秒`
      : rangeWaitingReturn
        ? "60 秒采集完成"
        : fixedPositioning
          ? "10 秒后自动开始 ID3–ID6"
          : "ID3–ID6 将自动采集 60 秒";
    if (neutralCountdownTimer !== null) {
      elements.neutralButton.textContent = `取消倒计时 · ${neutralCountdownRemaining} 秒`;
    } else {
      elements.neutralButton.textContent = state.neutral_collecting
        ? `正在采集 · ${Math.ceil(Number(state.neutral_remaining_s) || 0)} 秒`
        : "3 秒后开始采集 30 秒";
    }
    if (armCountdownTimer !== null) {
      elements.armButton.textContent = `取消 ARM · ${armCountdownRemaining} 秒`;
    } else {
      elements.armButton.textContent = pendingCommand === "/api/arm"
        ? "正在启动跟随…"
        : "ARM · 3 秒后开始跟随";
    }

    if (neutralCountdownTimer !== null) {
      showNeutralCountdownMessage();
    } else if (armCountdownTimer !== null) {
      showArmCountdownMessage();
    }

    if (gazeCalibrating) {
      [
        elements.baseCalibrationStartButton, elements.baseCalibrationMiddleButton,
        elements.baseCalibrationRangeStartButton, elements.baseCalibrationFinishButton,
        elements.fixedButton, elements.rangeFinishButton, elements.neutralButton, elements.armButton,
      ].forEach(button => { button.disabled = true; });
    }
    if (gazeProbing(state)) {
      [elements.connectButton, elements.followerPort, elements.disconnectButton]
        .forEach(button => { button.disabled = true; });
    }
    renderCompanion(state);
    renderJoints(Array.isArray(state.joints) ? state.joints : []);
    elements.connectionState.textContent = `后台在线 · 客户端 ${clientId.slice(0, 8)}`;

    if (elements.cameraForm) {
      renderCamera(state);
    } else if (!videoFailed && previewState === "RUNNING") {
      elements.videoFallback.hidden = true;
      elements.videoFeed.hidden = false;
    }
    window.dispatchEvent(new Event("so101:follower-state"));
  }

  function cameraViewOpen() {
    return elements.videoDialog.open || elements.companionWorkspaceDialog.open;
  }

  function mountCameraPanel(slot) {
    slot.append(elements.cameraPanel);
    elements.cameraPanelTitle.textContent = elements.companionWorkspaceDialog.open
      ? "腕部相机画面" : "实时人体与手部姿态";
    cameraRetainedForCompanion = false;
    disconnectCameraAfterConnect = false;
    void refreshCameraDevices();
    if (currentState) {
      renderCompanion(currentState);
      renderCamera(currentState);
    }
  }

  function closeCameraView() {
    if (cameraViewOpen()) return;
    detachVideo();
    if (!currentState?.operator) return;
    if (gazeCalibrationActive(currentState)) void requestGazeCalibration("cancel");
    if (currentState.companion?.enabled || companionStart) {
      cameraRetainedForCompanion = true;
    } else if (cameraCommand === "connect") {
      disconnectCameraAfterConnect = true;
    } else if (cameraIsActive(currentState)) {
      void requestCamera("disconnect");
    }
    renderCompanion(currentState);
  }

  function cameraIsActive(state) {
    return ["STARTING", "RUNNING"].includes(state?.preview_state);
  }

  function wristCameraName(state) {
    return state?.companion?.camera_name || DEFAULT_WRIST_CAMERA;
  }

  function wristCameraSelected(state) {
    return state.camera_name?.trim().toLowerCase() === wristCameraName(state).toLowerCase();
  }

  function gazeCalibrationActive(state) {
    return Boolean(state?.companion?.gaze?.calibrating || gazeCommand === "start");
  }

  function gazePreparing(state) {
    const gaze = state?.companion?.gaze;
    return Boolean(gaze?.calibrating && (gaze.preparing ||
      ["waiting", "preparing"].includes(gaze.state) || ["waiting", "preparing"].includes(gaze.phase)));
  }

  function gazeProbing(state) {
    return Boolean(state?.companion?.gaze?.calibrating) && !gazePreparing(state);
  }

  function gazeRawCalibrationDetails(gaze) {
    const diagnostics = gaze?.diagnostics || {};
    const positions = diagnostics.raw_positions || {};
    const limits = diagnostics.raw_limits || {};
    const clipped = Array.isArray(diagnostics.clipped_joints)
      ? diagnostics.clipped_joints.filter(joint => typeof joint === "string") : [];
    const joints = clipped.length ? clipped : [...new Set([...Object.keys(positions), ...Object.keys(limits)])];
    const labels = {
      shoulder_pan: "底座", shoulder_lift: "肩部", elbow_flex: "肘部",
      wrist_flex: "腕部俯仰", wrist_roll: "腕部旋转", gripper: "夹爪",
    };
    const lookup = (values, joint) => values[joint] ?? values[joint.replace(/\.pos$/, "")] ?? values[`${joint}.pos`];
    const number = value => typeof value === "number" && Number.isFinite(value) ? String(value) : "未读取";
    return joints.map(joint => {
      const record = lookup(limits, joint);
      const low = Array.isArray(record) ? record[0] : record?.min ?? record?.range_min ?? record?.low;
      const high = Array.isArray(record) ? record[1] : record?.max ?? record?.range_max ?? record?.high;
      const label = labels[joint.replace(/\.pos$/, "")] || joint;
      return `${label}：原始位置 ${number(lookup(positions, joint))}，校准记录 [${number(low)}, ${number(high)}]`;
    }).join("；");
  }

  function renderGazeCalibration(state) {
    const gaze = state.companion?.gaze;
    const active = gazeCalibrationActive(state);
    const preparing = gazePreparing(state);
    const countdownLabel = active && gaze?.phase === "countdown"
      ? `${Math.max(0, Math.ceil(Number(gaze.countdown_remaining) || 0))}秒后开始找脸` : "";
    const calibrationIssue = {
      calibration_range_mismatch: "需检查舵机校准",
      calibration_read_failed: "无法读取舵机位置",
    }[gaze?.phase];
    const issueReason = calibrationIssue && (gaze.reason || (gaze.phase === "calibration_range_mismatch"
      ? "当前姿态超出舵机校准记录，需要先检查舵机校准。" : "无法读取原始舵机位置，未开始探测。"));
    const labels = {
      uncalibrated: "未标定", ready: "已标定", calibrating: "标定中", waiting: "等待准备", preparing: "等待准备",
      fault: gaze?.calibrated ? "注视已暂停" : "标定未完成", pose_changed: "姿态变化", camera_changed: "相机变化",
      holding: "保持中", tracking: "正在纠偏", centered: "已居中",
    };
    const needsAttention = Boolean(calibrationIssue) || ["fault", "pose_changed", "camera_changed"].includes(gaze?.state);
    const activeLabel = gazeCommand === "start" && !gaze?.calibrating ? "正在启动" : countdownLabel ? "准备开始" : preparing ? "等待准备" : "标定中";
    setBadge(elements.gazeCalibrationBadge, calibrationIssue || (active ? activeLabel : labels[gaze?.state] || "未标定"),
      needsAttention ? "warning" : gaze?.calibrated ? "safe" : "muted");
    let blockedReason = "";
    if (!gaze) blockedReason = "正在读取注视标定状态，请确认本地服务已更新。";
    else if (!state.operator) blockedReason = "请先取得本页的机械臂控制权。";
    else if (state.companion.enabled) blockedReason = "请先暂停陪伴，再进行注视方向标定。";
    else if (state.armed || state.agent_motion?.active || state.teach_active || state.neutral_collecting ||
        state.release_pending || state.base_calibration?.active ||
        ["POSITIONING_FIXED", "STARTING", "RECORDING", "WAITING_RETURN", "FINISHING"].includes(state.safe_range?.state) ||
        neutralCountdownTimer !== null || armCountdownTimer !== null) blockedReason = "请先结束当前机械臂操作，再进行标定。";
    else if (busy || companionStart) blockedReason = "等待当前操作完成后开始标定。";
    elements.gazeCalibrationStartButton.disabled = active || Boolean(gazeCommand) || Boolean(blockedReason);
    elements.gazeCalibrationStartButton.title = active ? "正在进行标定" : blockedReason;
    const lastFailure = gazeCommandError || gaze?.save_error || (gaze?.state === "fault" ? gaze.reason : "");
    elements.gazeCalibrationStartButton.textContent = gazeCommand === "start" ? "正在启动标定…" :
      active ? countdownLabel || (preparing ? "正在准备…" : "正在标定…") :
      gaze?.calibrated ? "重新标定注视方向" : lastFailure || needsAttention ? "重试注视标定" : "开始注视标定";
    elements.gazeCalibrationCancelButton.disabled = !active || gazeCommand === "cancel" ||
      !state.operator || state.release_pending;
    elements.gazeCalibrationCancelButton.hidden = !active;
    elements.gazeCalibrationCancelButton.textContent = gazeCommand === "cancel" ? "正在取消…" : "取消标定";
    const progress = Math.round(Math.min(1, Math.max(0, Number(gaze?.progress) || 0)) * 100);
    elements.gazeCalibrationProgress.hidden = !active || preparing || Boolean(calibrationIssue);
    elements.gazeCalibrationProgress.value = progress;
    elements.gazeCalibrationProgress.setAttribute("aria-valuenow", String(progress));
    const message = issueReason || (active
      ? countdownLabel || gaze?.reason || (preparing ? "等你入镜。" : "正在标定，请保持自然姿态。")
      : blockedReason || (lastFailure && !gaze?.calibrated
        ? "上次未完成，可以重试。"
        : needsAttention ? gaze.reason || "标定已失效，请重新标定注视方向。"
        : gaze?.calibrated ? gaze.reason || "注视方向已标定，可以开始陪伴。"
        : "准备好后点击开始。"));
    elements.gazeCalibrationStatus.textContent = message;
    elements.gazeCalibrationStatus.classList.toggle("is-error", Boolean(calibrationIssue) || (!active && (Boolean(blockedReason && state.error) || (!blockedReason && needsAttention))));
    const rawDetails = calibrationIssue ? gazeRawCalibrationDetails(gaze) : "";
    const history = calibrationIssue ? [issueReason, rawDetails].filter(Boolean).join(" ") : lastFailure || "";
    elements.gazeCalibrationLastErrorDetails.hidden = !history;
    elements.gazeCalibrationLastErrorSummary.textContent = calibrationIssue ? "查看舵机校准检查记录" : "查看上次标定未完成的原因";
    elements.gazeCalibrationLastError.textContent = history;
    const validError = Array.isArray(gaze?.error) && gaze.error.length === 2 && gaze.error.every(Number.isFinite);
    elements.gazeCenterError.hidden = !validError;
    if (validError) {
      const [x, y] = gaze.error;
      elements.gazeCenterError.textContent = `人脸距中央：水平 ${Math.abs(x * 100).toFixed(1)}%${x < 0 ? "（左）" : x > 0 ? "（右）" : ""} · 垂直 ${Math.abs(y * 100).toFixed(1)}%${y < 0 ? "（上）" : y > 0 ? "（下）" : ""}`;
    }
  }

  function renderCompanion(state) {
    const companion = state.companion;
    const gazeCalibrating = gazeCalibrationActive(state);
    const gazeReady = Boolean(companion?.gaze?.calibrated) && !gazeCalibrating;
    const enabled = Boolean(companion?.enabled);
    const wristName = wristCameraName(state);
    const cameraActive = cameraIsActive(state);
    const wrongCamera = cameraActive && !wristCameraSelected(state);
    const releasingCamera = state.preview_state === "STOPPING" || cameraCommand === "disconnect";
    const robotFault = state.robot_state === "FAULT" || (state.robot_connected && Boolean(state.error));
    const robotReady = state.robot_connected && state.robot_state === "CONNECTED_DISARMED";
    const rangeState = state.safe_range?.state;
    const robotBusy = state.neutral_collecting || state.release_pending || state.agent_motion?.active ||
      state.teach_active || state.base_calibration?.active || (rangeState && rangeState !== "IDLE") ||
      neutralCountdownTimer !== null || armCountdownTimer !== null || gazeCalibrating;
    const writable = Boolean(companion && state.operator && state.robot_allowed && !busy && !companionStart);
    elements.companionFollowerPort.value = elements.followerPort.value;
    elements.companionFollowerPort.disabled = elements.followerPort.disabled || Boolean(companionStart);
    elements.companionConnectButton.disabled = elements.connectButton.disabled || Boolean(companionStart);
    elements.companionDisconnectButton.disabled = elements.disconnectButton.disabled || Boolean(companionStart) || gazeProbing(state);
    elements.companionConnectButton.textContent = state.robot_state === "CONNECTING" ? "正在连接…" :
      state.robot_connected ? "已连接机械臂" : "连接机械臂";
    const connectionMessage = state.robot_state === "CONNECTING" ? "正在连接从臂，请稍候…" :
      state.error ? state.error : state.robot_connected ? `从臂已连接 · ${state.robot_port || state.port}` :
      "从臂未连接。";
    elements.companionRobotStatus.textContent = connectionMessage;
    elements.companionRobotStatus.classList.toggle("is-error", Boolean(state.error));
    const labels = {
      off: "已关闭",
      waiting_camera: "等待相机",
      camera_required: "需要腕部相机",
      ambiguous: "等待单人",
      checking_presence: "确认在场",
      waiting_person: "等待入镜",
      waiting_face: "等待定位头部",
      waiting_motion: "等待动作结束",
      searching: companion?.allow_vertical_search ? "左右、上下找头" : "左右找头",
      search_blocked: "搜索受限",
      range_blocked: "部分关节受限",
      too_close: "距离过近 · 已暂停",
      cooldown: "等待对话动作",
      moving: "陪伴动作进行中",
    };
    const status = companion?.state;
    const offReason = !enabled && companion?.reason !== "陪伴模式已关闭。" ? companion?.reason : "";
    const fault = Boolean(offReason && /^(陪伴动作已停止|陪伴动作发送失败)/.test(offReason));
    const label = robotFault ? "从臂故障" : enabled ? labels[status] || "已开启" : fault ? "陪伴已停止" : labels.off;
    const warning = robotFault || fault ||
      ["waiting_camera", "camera_required", "ambiguous", "too_close", "search_blocked", "range_blocked"].includes(status);
    setBadge(elements.companionBadge, label, warning ? "warning" : enabled ? "safe" : "muted");

    const configuring = writable && !enabled && !gazeCalibrating;
    elements.companionUnoccupied.disabled = !configuring;
    if (enabled) {
      elements.companionUnoccupied.checked = Boolean(companion.allow_unoccupied_idle);
    } else if (companionWasEnabled) {
      elements.companionUnoccupied.checked = false;
    }
    companionWasEnabled = enabled;
    elements.companionToggleButton.disabled = !writable || (!enabled && (
      !robotReady || robotFault || robotBusy || wrongCamera || releasingCamera || cameraCommand === "connect" || !gazeReady
    ));
    elements.companionEnterButton.textContent = enabled ? "查看陪伴" : "开启陪伴";
    elements.companionToggleButton.textContent = companionStart ? "正在准备…" : enabled ? "暂停陪伴" : "开始陪伴";

    let message;
    if (!companion) message = "等待本地陪伴服务就绪。";
    else if (!state.robot_allowed) message = "当前是预览模式。使用真机模式启动后可连接从臂。";
    else if (robotFault) {
      const recover = state.can_disconnect ? "请先断开从臂，再在此窗口重新连接。" : "请重新连接从臂。";
      message = `从臂故障。 ${state.error || offReason || ""} ${recover}`.replace(/\s+/g, " ").trim();
    } else if (!state.operator) message = "请先取得本页的机械臂控制权。";
    else if (gazeCalibrating) message = gazePreparing(state)
      ? companion.gaze.reason || "等你入镜，完成标定后可开始陪伴。"
      : "正在标定，完成后可开始陪伴。";
    else if (companionStart) message = `正在准备腕部相机 ${wristName}，画面就绪后开始陪伴…`;
    else if (enabled) message = companion.reason || "陪伴模式已开启。";
    else if (companionCommandError) message = companionCommandError;
    else if (state.armed) message = "结束人体跟随后可开启陪伴模式。";
    else if (robotBusy) message = "等待当前操作完成。";
    else if (!robotReady) message = "请在本界面填写串口并连接机械臂。";
    else if (wrongCamera) message = `请断开当前相机。请连接腕部相机 ${wristName}。`;
    else if (releasingCamera) message = "正在释放相机，请稍候。";
    else if (fault) message = offReason;
    else if (!gazeReady) message = companion.gaze?.reason || "请先连接腕部相机，在「注视方向标定」中完成标定，再开始陪伴。";
    else if (!cameraActive) message = `点击「开始陪伴」会自动连接腕部相机 ${wristName}。`;
    else message = "点击「开始陪伴」即可开始。";
    if (elements.companionStatus.textContent !== message) elements.companionStatus.textContent = message;
    elements.companionStatus.classList.toggle("is-error", robotFault || fault || message === companionCommandError);
    elements.companionToggleButton.title = elements.companionToggleButton.disabled ? message : "";
    renderGazeCalibration(state);

    if (cameraRetainedForCompanion && !enabled && !companionStart && !cameraViewOpen()) {
      cameraRetainedForCompanion = false;
      if (state.operator && cameraIsActive(state)) {
        queueMicrotask(() => {
          if (!cameraViewOpen() && !currentState?.companion?.enabled && !companionStart) void requestCamera("disconnect");
        });
      }
    }
  }

  function detachVideo() {
    elements.videoFeed.removeAttribute("src");
    elements.videoFeed.hidden = true;
    elements.videoFallback.hidden = false;
    videoFailed = false;
    videoSession = null;
  }

  function renderCamera(state) {
    const preview = state?.preview_state || "STOPPED";
    const active = cameraIsActive(state);
    const stopping = preview === "STOPPING" || cameraCommand === "disconnect";
    const starting = preview === "STARTING" || cameraCommand === "connect";
    const writable = Boolean(state?.operator) && !cameraCommand && !gazeProbing(state);
    const wristName = wristCameraName(state);
    if (elements.cameraName.defaultValue !== wristName) elements.cameraName.defaultValue = wristName;
    const name = (cameraCommand === "connect" ? elements.cameraName.value.trim() : state?.camera_name) || "默认相机";
    elements.cameraName.disabled = !writable || active || stopping;
    elements.cameraConnectButton.disabled = elements.cameraName.disabled;
    elements.cameraDisconnectButton.disabled = !writable || !active || stopping;
    elements.cameraConnectButton.textContent = starting ? "正在连接…" : "连接视频";
    elements.cameraDisconnectButton.textContent = stopping ? "正在断开…" : "断开视频";

    const closeHint = state?.companion?.enabled || companionStart
      ? "关闭此窗口后，陪伴模式会继续使用相机。"
      : "关闭此窗口也会断开相机。";
    const status = stopping ? "正在释放相机…" : starting ? `正在打开 ${name}…` :
      preview === "RUNNING" ? `${name} 已连接。${state?.operator ? closeHint : ""}` :
      `陪伴模式使用腕部相机 ${wristName}。可填写设备管理器中的其他相机名称用于人体跟随。`;
    const error = cameraCommandError || (preview === "FAULT" ? state?.preview_error || "相机无法连接。" : "");
    if (elements.cameraStatus.textContent !== (error || status)) elements.cameraStatus.textContent = error || status;
    elements.cameraStatus.classList.toggle("is-error", Boolean(error));

    if (cameraViewOpen() && preview === "RUNNING" && !stopping) {
      const session = String(state.camera_session ?? "current");
      if (videoSession !== session || !elements.videoFeed.hasAttribute("src")) {
        videoFailed = false;
        videoSession = session;
        elements.videoFeed.src = `/video_feed?session=${encodeURIComponent(session)}`;
      }
      elements.videoFeed.hidden = videoFailed;
      elements.videoFallback.hidden = !videoFailed;
      elements.videoFallbackTitle.textContent = "画面加载失败，请断开后重连";
      return;
    }
    detachVideo();
    elements.videoFallbackTitle.textContent = stopping ? "正在断开视频…" :
      starting ? "正在等待相机画面…" : preview === "FAULT" ? "相机连接失败" : "相机未连接";
  }

  function renderJoints(joints) {
    if (!joints.length) {
      elements.jointRows.replaceChildren(createCellRow("尚无舵机状态"));
      return;
    }

    const fragment = document.createDocumentFragment();
    joints.forEach((joint) => {
      const row = document.createElement("tr");
      row.className = joint.role === "fixed" ? "joint-fixed" : "joint-moving";

      const idCell = document.createElement("td");
      const idBadge = document.createElement("span");
      idBadge.className = "joint-id";
      idBadge.textContent = `ID ${joint.id}`;
      idCell.append(idBadge);

      const nameCell = document.createElement("td");
      const code = document.createElement("code");
      code.textContent = joint.name || "—";
      nameCell.append(code);

      const roleCell = document.createElement("td");
      const role = document.createElement("span");
      role.className = `role-pill ${joint.role === "fixed" ? "role-fixed" : "role-moving"}`;
      role.textContent = joint.role === "fixed" ? "视频固定 · Agent可用" : "视频跟随 · Agent可用";
      roleCell.append(role);

      const mappingCell = document.createElement("td");
      mappingCell.textContent = translateMapping(joint.mapping);

      const currentCell = document.createElement("td");
      currentCell.className = "number-cell";
      currentCell.textContent = formatValue(joint.current);

      const rangeCell = document.createElement("td");
      rangeCell.className = "number-cell";
      if (joint.role === "fixed") {
        rangeCell.textContent = joint.fixed_target === null || joint.fixed_target === undefined
          ? "未确认"
          : `固定 ${formatValue(joint.fixed_target)}`;
      } else if (
        joint.safe_min !== null && joint.safe_min !== undefined &&
        joint.safe_max !== null && joint.safe_max !== undefined
      ) {
        rangeCell.textContent = `${formatValue(joint.safe_min)} ～ ${formatValue(joint.safe_max)}`;
      } else {
        rangeCell.textContent = "未采集";
      }

      const startupCell = document.createElement("td");
      startupCell.className = "number-cell";
      startupCell.textContent = formatValue(joint.startup);

      const commandCell = document.createElement("td");
      commandCell.className = "number-cell";
      commandCell.textContent = formatValue(joint.last_command);

      row.append(
        idCell,
        nameCell,
        roleCell,
        mappingCell,
        currentCell,
        rangeCell,
        startupCell,
        commandCell,
      );
      fragment.append(row);
    });
    elements.jointRows.replaceChildren(fragment);
  }

  function translateMapping(mapping) {
    const labels = {
      "startup hold": "保持启动位置",
      fixed_at_startup: "保持启动位置",
      "human up/down": "人体手臂上下",
      "human left/right": "人体手臂左右",
      "human elbow bend": "人体肘部弯曲",
      "human arm lift": "手臂抬高 / 降低",
      human_arm_lift: "手臂抬高 / 降低",
      human_elbow_bend: "人体肘部弯曲",
      "human hand roll": "手掌旋转",
      "human palm roll": "手掌旋转",
      hand_roll: "手掌旋转",
      "human hand open": "张手 / 握拳",
      "human hand openness": "张手 / 握拳",
      hand_open: "张手 / 握拳",
    };
    return labels[mapping] || mapping || "—";
  }

  function createCellRow(message) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.className = "loading-cell";
    cell.textContent = message;
    row.append(cell);
    return row;
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-Client-Id", clientId);
    if (options.body) headers.set("Content-Type", "application/json");
    const response = await fetch(path, {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
      headers,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || payload.message || `请求失败：${response.status}`);
    }
    return payload;
  }

  async function refreshCameraDevices() {
    try {
      const { devices } = await api("/api/camera/devices");
      if (!Array.isArray(devices)) return;
      const names = new Set(devices.map((device) => device?.name).filter((name) => typeof name === "string" && name));
      elements.cameraDevices.replaceChildren(...[...names].map((name) => new Option(name)));
    } catch (_error) {
      // Device suggestions are optional. The name can still be entered manually.
    }
  }

  async function requestCamera(action, name = elements.cameraName.value.trim()) {
    if (!currentState?.operator || cameraCommand) return;
    cameraCommand = action;
    cameraCommandError = "";
    if (action === "disconnect") {
      companionStart?.abort();
      cancelNeutralCountdown(false);
      cancelArmCountdown(false);
    }
    render(currentState);
    try {
      const payload = action === "connect" ? { name } : {};
      const result = await api(`/api/camera/${action}`, { method: "POST", body: JSON.stringify(payload) });
      if (result.preview_state) {
        currentState = {
          ...currentState,
          preview_state: result.preview_state,
          preview_error: result.preview_error,
          camera_name: result.camera_name,
          camera_session: result.camera_session ?? currentState.camera_session,
        };
      }
    } catch (error) {
      cameraCommandError = error.message;
    } finally {
      cameraCommand = null;
      render(currentState);
      if (action === "connect" && disconnectCameraAfterConnect) {
        disconnectCameraAfterConnect = false;
        if (!cameraViewOpen()) void requestCamera("disconnect");
      }
      void pollStatus(true);
    }
    return cameraCommandError;
  }

  async function pollStatus(forceFresh = false) {
    if (pollingPromise) {
      if (!forceFresh) return pollingPromise;
      await pollingPromise;
    }
    const request = (async () => {
    try {
      const state = await api("/api/status");
      render(state);
      return state;
    } catch (error) {
      cancelNeutralCountdown(false);
      cancelArmCountdown(false);
      elements.connectionState.textContent = "后台连接中断";
      elements.globalMessage.textContent = `无法读取后台状态：${error.message}`;
      elements.globalMessage.classList.add("is-error");
      disableAllControls();
      if (elements.cameraForm) {
        detachVideo();
        elements.videoFallbackTitle.textContent = "无法读取相机状态";
      }
      window.dispatchEvent(new Event("so101:follower-state"));
    }
    })();
    pollingPromise = request;
    try {
      return await request;
    } finally {
      if (pollingPromise === request) pollingPromise = null;
    }
  }

  async function runCommand(path, body = null) {
    if (busy) return;
    busy = true;
    pendingCommand = path;
    let commandError = null;
    disableAllControls();
    if ((path === "/api/arm" || path === "/api/neutral" || companionStart) &&
      currentState?.operator && !currentState.release_pending) {
      elements.stopButton.disabled = false;
    }
    try {
      const options = { method: "POST" };
      if (body !== null) options.body = JSON.stringify(body);
      const result = await api(path, options);
      elements.globalMessage.textContent = result.message || "命令已执行";
      await pollStatus(true);
    } catch (error) {
      commandError = error.message;
    } finally {
      busy = false;
      pendingCommand = null;
      if (currentState) render(currentState);
      if (commandError) {
        elements.globalMessage.textContent = commandError;
        elements.globalMessage.classList.add("is-error");
      }
    }
    return commandError;
  }

  function cancelPendingStarts() {
    cancelNeutralCountdown(false);
    cancelArmCountdown(false);
    companionStart?.abort();
  }

  async function stopFollowing() {
    if (!currentState?.operator) return;
    cancelPendingStarts();
    elements.stopButton.disabled = true;
    elements.globalMessage.textContent = "正在停止机械臂动作…";
    try {
      const sequence = window.so101InvalidateInteraction?.();
      const result = await api("/api/disarm", { method: "POST", body: JSON.stringify({ client_sequence: sequence }) });
      await pollStatus(true);
      elements.globalMessage.textContent = result.message || "已停止机械臂动作并保持扭矩";
      elements.globalMessage.classList.remove("is-error", "is-armed");
    } catch (error) {
      elements.globalMessage.textContent = error.message;
      elements.globalMessage.classList.add("is-error");
    }
  }

  async function waitForCompanionCamera(epoch, signal) {
    const deadline = performance.now() + COMPANION_CAMERA_TIMEOUT_MS;
    let state = await pollStatus(true);
    while (performance.now() < deadline) {
      signal.throwIfAborted();
      if (!state) throw new Error("无法读取本地控制服务状态，请重新开启陪伴。");
      if (!state.operator || state.motion_epoch !== epoch) throw new DOMException("已取消陪伴启动", "AbortError");
      if (state.preview_state === "FAULT") throw new Error(state.preview_error || "腕部相机连接失败。");
      if (!cameraIsActive(state)) throw new Error("腕部相机已断开，请重新开启陪伴。");
      if (!wristCameraSelected(state)) throw new Error("当前相机已切换，请连接腕部相机。");
      if (state.companion?.camera_ready) return;
      await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS));
      state = await pollStatus();
    }
    throw new Error("等待腕部相机画面超时，请查看视频连接。");
  }

  async function toggleCompanion() {
    if (elements.companionToggleButton.disabled) return;
    companionCommandError = "";
    if (currentState.companion.enabled) {
      companionCommandError = await runCommand("/api/companion", { enabled: false }) || "";
      if (currentState) render(currentState);
      return;
    }

    // Calibrating is a separate, explicit motion request. Starting companion
    // behavior must never silently probe the joints or assume motor directions.
    if (!currentState.companion.gaze?.calibrated || gazeCalibrationActive(currentState)) {
      companionCommandError = "请先完成「注视方向标定」，再开始陪伴。";
      render(currentState);
      return;
    }

    const epoch = currentState.motion_epoch;
    const allowUnoccupiedIdle = elements.companionUnoccupied.checked;
    const start = new AbortController();
    companionStart = start;
    if (!cameraViewOpen()) cameraRetainedForCompanion = true;
    render(currentState);
    try {
      start.signal.throwIfAborted();
      if (!cameraIsActive(currentState)) {
        const name = wristCameraName(currentState);
        elements.cameraName.value = name;
        const error = await requestCamera("connect", name);
        if (error) throw new Error(error);
      }
      await waitForCompanionCamera(epoch, start.signal);
      start.signal.throwIfAborted();
      if (busy) throw new Error("等待当前操作完成后，请重新开启陪伴。");
      const error = await runCommand("/api/companion", {
        enabled: true, allow_unoccupied_idle: allowUnoccupiedIdle,
        allow_vertical_search: allowUnoccupiedIdle, expected_epoch: epoch,
      });
      if (error && !start.signal.aborted) companionCommandError = error;
    } catch (error) {
      if (!start.signal.aborted && error.name !== "AbortError") companionCommandError = error.message;
    } finally {
      companionStart = null;
      if (currentState) render(currentState);
    }
  }

  function disableAllControls() {
    [
      elements.followerPort,
      elements.connectButton,
      elements.baseCalibrationStartButton,
      elements.baseCalibrationMiddleButton,
      elements.baseCalibrationRangeStartButton,
      elements.baseCalibrationFinishButton,
      elements.baseCalibrationCancelButton,
      elements.fixedButton,
      elements.rangeStartButton,
      elements.rangeFinishButton,
      elements.rangeCancelButton,
      elements.neutralButton,
      elements.armButton,
      elements.stopButton,
      elements.disconnectButton,
      elements.cameraName,
      elements.cameraConnectButton,
      elements.cameraDisconnectButton,
      elements.companionToggleButton,
      elements.companionUnoccupied,
      elements.companionFollowerPort,
      elements.companionConnectButton,
      elements.companionDisconnectButton,
      elements.gazeCalibrationStartButton,
      elements.gazeCalibrationCancelButton,
    ].forEach((button) => { if (button) button.disabled = true; });
  }

  async function requestGazeCalibration(action) {
    if (!currentState?.operator) return;
    if (action === "start" && (elements.gazeCalibrationStartButton.disabled || gazeCommand)) return;
    if (action === "cancel" && !(currentState.companion?.gaze?.calibrating || gazeCommand === "start")) return;
    const sequence = ++gazeRequestSequence;
    gazeCommand = action;
    gazeCommandError = "";
    companionCommandError = "";
    render(currentState);
    try {
      const result = await api("/api/gaze/calibrate", {
        method: "POST",
        body: JSON.stringify({
          action, expected_epoch: currentState.motion_epoch, confirmation: CONTROL_CONFIRMATION_CODE,
        }),
      });
      if (sequence === gazeRequestSequence && result.companion) {
        currentState = { ...currentState, companion: result.companion };
      }
      await pollStatus(true);
    } catch (error) {
      if (sequence === gazeRequestSequence) gazeCommandError = error.message;
    } finally {
      if (sequence === gazeRequestSequence) gazeCommand = null;
      if (currentState) render(currentState);
    }
  }

  function showNeutralCountdownMessage() {
    elements.globalMessage.textContent =
      `中立姿势将在 ${neutralCountdownRemaining} 秒后开始连续采集 30 秒，请走到摄像头前并保持自然姿势。再次点击按钮可取消。`;
    elements.globalMessage.classList.remove("is-error", "is-armed");
  }

  function updateNeutralCountdownDisplay() {
    elements.neutralButton.disabled = false;
    elements.neutralButton.textContent = `取消倒计时 · ${neutralCountdownRemaining} 秒`;
    elements.neutralProgressText.textContent = `准备入镜 · ${neutralCountdownRemaining} 秒后开始采集`;
    showNeutralCountdownMessage();
  }

  function cancelNeutralCountdown(notify = true) {
    if (neutralCountdownTimer === null) return;
    window.clearInterval(neutralCountdownTimer);
    neutralCountdownTimer = null;
    neutralCountdownRemaining = 0;
    if (notify) {
      elements.globalMessage.textContent = "已取消中立采集倒计时。";
      elements.globalMessage.classList.remove("is-error", "is-armed");
    }
  }

  function startNeutralCountdown() {
    if (neutralCountdownTimer !== null) {
      cancelNeutralCountdown(true);
      if (currentState) render(currentState);
      elements.globalMessage.textContent = "已取消中立采集倒计时。";
      elements.globalMessage.classList.remove("is-error", "is-armed");
      return;
    }
    if (armCountdownTimer !== null || !neutralCountdownAllowed(currentState)) {
      elements.globalMessage.textContent = "机器人连接并保持 DISARMED 后才能安排中立采集。";
      elements.globalMessage.classList.add("is-error");
      return;
    }

    neutralCommandError = "";
    neutralCountdownRemaining = ACTION_COUNTDOWN_SECONDS;
    neutralCountdownTimer = window.setInterval(async () => {
      neutralCountdownRemaining -= 1;
      if (neutralCountdownRemaining > 0) {
        updateNeutralCountdownDisplay();
        return;
      }
      window.clearInterval(neutralCountdownTimer);
      neutralCountdownTimer = null;
      neutralCountdownRemaining = 0;
      if (busy || !neutralCountdownAllowed(currentState)) {
        neutralCommandError = "连接或控制状态已变化，请确认后重新采集。";
        if (currentState) render(currentState);
        elements.globalMessage.textContent = "状态已变化，中立采集已取消。";
        elements.globalMessage.classList.add("is-error");
        return;
      }
      elements.globalMessage.textContent = "正在开始 30 秒中立姿势采集…";
      elements.neutralProgressText.textContent = "正在启动 30 秒采集…";
      neutralCommandError = await runCommand("/api/neutral") || "";
      if (currentState) render(currentState);
    }, 1000);
    if (currentState) render(currentState);
    updateNeutralCountdownDisplay();
  }

  function showArmCountdownMessage() {
    if (armCountdownPhase === "waiting") {
      const reason = translatePoseReason(currentState?.pose_reason);
      elements.globalMessage.textContent =
        `等待手臂与手部识别稳定（当前：${reason}）。请让肩部、右肘、右腕和右手入镜；${armAcquireRemaining} 秒内有效，再自动开始 3 秒 ARM 倒计时。`;
    } else {
      elements.globalMessage.textContent =
        `人体识别已稳定，将在 ${armCountdownRemaining} 秒后 ARM。保持姿势；识别中断会重新等待。`;
    }
    elements.globalMessage.classList.remove("is-error", "is-armed");
  }

  function updateArmCountdownDisplay() {
    elements.armButton.disabled = false;
    elements.armButton.textContent = armCountdownPhase === "waiting"
      ? `取消 ARM · 等待识别 ${armAcquireRemaining} 秒`
      : `取消 ARM · ${armCountdownRemaining} 秒`;
    showArmCountdownMessage();
  }

  function cancelArmCountdown(notify = true) {
    if (armCountdownTimer === null) return;
    window.clearInterval(armCountdownTimer);
    armCountdownTimer = null;
    armCountdownRemaining = 0;
    armAcquireRemaining = 0;
    armCountdownPhase = "idle";
    if (notify) {
      elements.globalMessage.textContent = "已取消 ARM 倒计时，机械臂仍保持 DISARMED。";
      elements.globalMessage.classList.remove("is-error", "is-armed");
    }
  }

  function startArmCountdown() {
    if (
      neutralCountdownTimer !== null ||
      armCountdownTimer !== null ||
      !armCountdownSafe(currentState)
    ) {
      elements.globalMessage.textContent = "机器人连接或中立采集状态不满足 ARM 条件。";
      elements.globalMessage.classList.add("is-error");
      return;
    }

    armCountdownRemaining = ACTION_COUNTDOWN_SECONDS;
    armAcquireRemaining = ARM_ACQUIRE_TIMEOUT_SECONDS;
    armCountdownPhase = currentState?.can_arm ? "countdown" : "waiting";
    armCountdownTimer = window.setInterval(() => {
      if (!armCountdownSafe(currentState)) {
        cancelArmCountdown(false);
        if (currentState) render(currentState);
        elements.globalMessage.textContent = "连接或中立采集状态已变化，ARM 已取消；机械臂仍保持 DISARMED。";
        elements.globalMessage.classList.add("is-error");
        return;
      }

      armAcquireRemaining -= 1;
      if (armAcquireRemaining <= 0) {
        cancelArmCountdown(false);
        if (currentState) render(currentState);
        elements.globalMessage.textContent = "30 秒内没有稳定识别到手臂和手，ARM 已取消；请让肩部、右肘、右腕和右手进入镜头后重试。";
        elements.globalMessage.classList.add("is-error");
        return;
      }

      if (!currentState?.can_arm) {
        armCountdownPhase = "waiting";
        armCountdownRemaining = ACTION_COUNTDOWN_SECONDS;
        updateArmCountdownDisplay();
        return;
      }

      if (armCountdownPhase === "waiting") {
        armCountdownPhase = "countdown";
        armCountdownRemaining = ACTION_COUNTDOWN_SECONDS;
        updateArmCountdownDisplay();
        return;
      }

      armCountdownRemaining -= 1;
      if (armCountdownRemaining > 0) {
        updateArmCountdownDisplay();
        return;
      }

      window.clearInterval(armCountdownTimer);
      armCountdownTimer = null;
      armCountdownRemaining = 0;
      armAcquireRemaining = 0;
      armCountdownPhase = "idle";
      elements.globalMessage.textContent = "正在启动人体跟随…";
      runCommand("/api/arm", { confirmation: CONTROL_CONFIRMATION_CODE });
    }, 1000);
    if (currentState) render(currentState);
    updateArmCountdownDisplay();
  }

  function runFollowerCommand(operation) {
    const paths = {
      connect: "/api/connect",
      "calibration/start": "/api/base-calibration/start",
      "calibration/middle": "/api/base-calibration/middle",
      "calibration/range": "/api/base-calibration/range/start",
      "calibration/save": "/api/base-calibration/finish",
      "calibration/cancel": "/api/base-calibration/cancel",
    };
    const path = paths[operation];
    if (!path) return Promise.resolve("未知的从臂操作。");
    if (busy) return Promise.resolve("正在处理上一条机器人命令。");
    const port = elements.followerPort.value.trim();
    const needsPort = operation === "connect" || operation === "calibration/start";
    if (needsPort && !port) {
      const error = "请填写从臂串口。";
      elements.globalMessage.textContent = error;
      elements.globalMessage.classList.add("is-error");
      return Promise.resolve(error);
    }
    return runCommand(path, operation === "calibration/cancel" ? null : {
      confirmation: CONTROL_CONFIRMATION_CODE,
      ...(needsPort ? { port } : {}),
    });
  }

  function setFollowerPort(port) {
    if (elements.followerPort.disabled) return;
    elements.followerPort.value = port;
    elements.companionFollowerPort.value = port;
    followerPortEdited = true;
    window.dispatchEvent(new Event("so101:follower-state"));
  }

  window.so101FollowerControl = {
    get state() { return currentState; },
    get port() { return elements.followerPort.value; },
    get portDisabled() { return elements.followerPort.disabled; },
    cancelPendingStarts,
    setPort: setFollowerPort,
    run: runFollowerCommand,
  };
  elements.followerPort.addEventListener("input", () => setFollowerPort(elements.followerPort.value));

  elements.baseCalibrationStartButton.addEventListener("click", () => {
    void runFollowerCommand("calibration/start");
  });

  elements.baseCalibrationMiddleButton.addEventListener("click", () => {
    runCommand("/api/base-calibration/middle", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.baseCalibrationRangeStartButton.addEventListener("click", () => {
    runCommand("/api/base-calibration/range/start", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.baseCalibrationFinishButton.addEventListener("click", () => {
    runCommand("/api/base-calibration/finish", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.baseCalibrationCancelButton.addEventListener("click", () => {
    runCommand("/api/base-calibration/cancel");
  });

  elements.connectButton.addEventListener("click", () => {
    void runFollowerCommand("connect");
  });

  elements.fixedButton.addEventListener("click", () => {
    runCommand("/api/safe-range/fixed", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.rangeFinishButton.addEventListener("click", () => {
    runCommand("/api/safe-range/finish", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.rangeCancelButton.addEventListener("click", () => {
    runCommand("/api/safe-range/cancel");
  });

  elements.neutralButton.addEventListener("click", () => {
    startNeutralCountdown();
  });

  elements.armButton.addEventListener("click", () => {
    if (armCountdownTimer !== null) {
      cancelArmCountdown(true);
      if (currentState) render(currentState);
      elements.globalMessage.textContent = "已取消 ARM 倒计时，机械臂仍保持 DISARMED。";
      elements.globalMessage.classList.remove("is-error", "is-armed");
      return;
    }
    startArmCountdown();
  });

  elements.stopButton.addEventListener("click", () => {
    stopFollowing();
  });
  byId("openCompanionButton").addEventListener("click", () => {
    if (!elements.companionDialog.open) elements.companionDialog.showModal();
    if (currentState) renderCompanion(currentState);
  });
  elements.companionEnterButton.addEventListener("click", () => {
    elements.companionDialog.close();
    if (elements.videoDialog.open) elements.videoDialog.close();
    if (!elements.companionWorkspaceDialog.open) elements.companionWorkspaceDialog.showModal();
    mountCameraPanel(elements.companionCameraSlot);
    elements.companionWorkspaceDialog.dispatchEvent(new Event("so101:companion-open"));
  });
  byId("closeCompanionWorkspaceButton").addEventListener("click", () => elements.companionWorkspaceDialog.close());
  elements.companionWorkspaceDialog.addEventListener("click", event => {
    if (event.target === elements.companionWorkspaceDialog) elements.companionWorkspaceDialog.close();
  });
  byId("closeCompanionButton").addEventListener("click", () => elements.companionDialog.close());
  elements.companionDialog.addEventListener("click", event => {
    if (event.target === elements.companionDialog) elements.companionDialog.close();
  });
  elements.companionDialogStopButton.addEventListener("click", () => elements.stopButton.click());
  const syncCompanionStopAvailability = () => {
    elements.companionDialogStopButton.disabled = elements.stopButton.disabled;
  };
  syncCompanionStopAvailability();
  new MutationObserver(syncCompanionStopAvailability).observe(elements.stopButton, {
    attributes: true, attributeFilter: ["disabled"],
  });
  elements.companionFollowerPort.addEventListener("input", () => {
    setFollowerPort(elements.companionFollowerPort.value);
  });
  elements.companionRobotForm.addEventListener("submit", async event => {
    event.preventDefault();
    if (elements.companionConnectButton.disabled) return;
    companionCommandError = await runFollowerCommand("connect") || "";
    if (currentState) renderCompanion(currentState);
  });
  elements.companionDisconnectButton.addEventListener("click", async () => {
    if (elements.companionDisconnectButton.disabled) return;
    cancelPendingStarts();
    companionCommandError = await runCommand("/api/disconnect", { confirmation: CONTROL_CONFIRMATION_CODE }) || "";
    if (currentState) renderCompanion(currentState);
  });
  elements.companionToggleButton.addEventListener("click", () => void toggleCompanion());
  elements.gazeCalibrationStartButton.addEventListener("click", () => void requestGazeCalibration("start"));
  elements.gazeCalibrationCancelButton.addEventListener("click", () => void requestGazeCalibration("cancel"));
  byId("agentChatSettingsButton").addEventListener("click", () => byId("openAgentSettingsButton").click());

  elements.disconnectButton.addEventListener("click", () => {
    runCommand("/api/disconnect", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  if (elements.cameraForm) {
    elements.cameraName.addEventListener("input", () => {
      cameraCommandError = "";
      renderCamera(currentState);
    });
    elements.cameraForm.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!elements.cameraConnectButton.disabled) void requestCamera("connect");
    });
    elements.cameraDisconnectButton.addEventListener("click", () => void requestCamera("disconnect"));
    elements.videoDialog.addEventListener("so101:video-open", () => {
      if (elements.companionWorkspaceDialog.open) elements.companionWorkspaceDialog.close();
      mountCameraPanel(elements.videoCameraSlot);
    });
    elements.videoDialog.addEventListener("close", closeCameraView);
    elements.companionWorkspaceDialog.addEventListener("close", closeCameraView);
    window.addEventListener("pagehide", () => {
      if (!currentState?.operator || !(cameraIsActive(currentState) || cameraCommand === "connect")) return;
      void api("/api/camera/disconnect", { method: "POST", body: "{}", keepalive: true }).catch(() => {});
    });
  }

  elements.videoFeed.addEventListener("load", () => {
    if (elements.cameraForm && (!cameraViewOpen() || !elements.videoFeed.hasAttribute("src"))) return;
    videoFailed = false;
    elements.videoFallback.hidden = true;
    elements.videoFeed.hidden = false;
  });

  elements.videoFeed.addEventListener("error", () => {
    if (elements.cameraForm && (!cameraViewOpen() || !elements.videoFeed.hasAttribute("src"))) return;
    videoFailed = true;
    elements.videoFeed.hidden = true;
    elements.videoFallback.hidden = false;
  });

  function disarmOnLeave() {
    companionStart?.abort();
    cancelNeutralCountdown(false);
    cancelArmCountdown(false);
    // Send this even while the last known state is DISARMED: it also invalidates an
    // ARM request that may still be queued on the server.
    if (!currentState?.operator) return;
    if (currentState.base_calibration?.active) {
      fetch("/api/base-calibration/cancel", {
        method: "POST",
        headers: { "X-Client-Id": clientId },
        credentials: "same-origin",
        keepalive: true,
        cache: "no-store",
      }).catch(() => {});
    }
    if (["POSITIONING_FIXED", "STARTING", "RECORDING", "WAITING_RETURN", "FINISHING"]
      .includes(currentState.safe_range?.state)) {
      fetch("/api/safe-range/cancel", {
        method: "POST",
        headers: { "X-Client-Id": clientId },
        credentials: "same-origin",
        keepalive: true,
        cache: "no-store",
      }).catch(() => {});
    }
    fetch("/api/disarm", {
      method: "POST",
      headers: { "X-Client-Id": clientId, "Content-Type": "application/json" },
      body: JSON.stringify({ client_sequence: window.so101InvalidateInteraction?.() }),
      credentials: "same-origin",
      keepalive: true,
      cache: "no-store",
    }).catch(() => {});
  }

  window.addEventListener("pagehide", disarmOnLeave, { capture: true });
  window.addEventListener("beforeunload", disarmOnLeave, { capture: true });

  pollStatus();
  window.setInterval(pollStatus, POLL_INTERVAL_MS);
})();
