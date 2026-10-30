(() => {
  "use strict";

  const CLIENT_KEY = "so101-human-follow-client-id";
  const POLL_INTERVAL_MS = 500;
  const ACTION_COUNTDOWN_SECONDS = 3;
  const ARM_ACQUIRE_TIMEOUT_SECONDS = 30;
  const CONTROL_CONFIRMATION_CODE = "123";
  const clientId = getClientId();
  let currentState = null;
  let polling = false;
  let busy = false;
  let pendingCommand = null;
  let videoFailed = false;
  let neutralCountdownTimer = null;
  let neutralCountdownRemaining = 0;
  let armCountdownTimer = null;
  let armCountdownRemaining = 0;
  let armAcquireRemaining = 0;
  let armCountdownPhase = "idle";

  const elements = {
    globalMessage: byId("globalMessage"),
    operatorBadge: byId("operatorBadge"),
    armBadge: byId("armBadge"),
    previewBadge: byId("previewBadge"),
    robotBadge: byId("robotBadge"),
    readOnlyNotice: byId("readOnlyNotice"),
    videoFeed: byId("videoFeed"),
    videoFallback: byId("videoFallback"),
    poseDot: byId("poseDot"),
    poseSummary: byId("poseSummary"),
    frameAge: byId("frameAge"),
    poseReason: byId("poseReason"),
    peopleCount: byId("peopleCount"),
    validStreak: byId("validStreak"),
    sendAge: byId("sendAge"),
    robotIdentity: byId("robotIdentity"),
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
    sessionStorage.setItem(CLIENT_KEY, id);
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
      !state.armed &&
      !state.neutral_collecting &&
      !state.release_pending &&
      (!state.robot_allowed || state.safe_range?.profile_ready) &&
      !state.error
    );
  }

  function armCountdownSafe(state) {
    return Boolean(
      state?.operator &&
      state.robot_state === "CONNECTED_DISARMED" &&
      state.preview_state === "RUNNING" &&
      state.neutral_ready &&
      !state.armed &&
      !state.neutral_collecting &&
      !state.release_pending &&
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
    elements.neutralStepIndex.textContent = "3";
    elements.armStepIndex.textContent = "4";
    let countdownCancelled = false;
    if (neutralCountdownTimer !== null && !canScheduleNeutral) {
      cancelNeutralCountdown(false);
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
    setBadge(elements.previewBadge, previewState, badgeKind(previewState));
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

    elements.poseDot.classList.toggle("is-good", poseValid);
    elements.poseDot.classList.toggle("is-bad", !poseValid);
    const channelLabels = {
      lift: "手臂抬高",
      elbow: "肘部弯曲",
      hand_roll: "手掌旋转",
      hand_open: "张手/握拳",
    };
    const validChannels = Array.isArray(state.valid_channels) ? state.valid_channels : [];
    elements.poseSummary.textContent = validChannels.length
      ? `有效：${validChannels.map((name) => channelLabels[name] || name).join("、")}`
      : "暂无有效控制通道";
    elements.frameAge.textContent = `帧延迟 ${formatAge(state.frame_age_s)}`;
    elements.poseReason.textContent = translatePoseReason(state.pose_reason);
    elements.peopleCount.textContent = state.people_count ?? "—";
    elements.validStreak.textContent = state.valid_streak ?? "—";
    elements.sendAge.textContent = formatAge(state.last_send_age_s);

    elements.robotIdentity.textContent = `${state.robot_id || "SO-101"} · ${state.port || "—"}`;
    elements.leaseState.textContent = operator
      ? `本页面 · 心跳 ${formatAge(state.lease_age_s)}`
      : `只读 · ${formatAge(state.lease_age_s)}`;
    elements.robotState.textContent = robotState;
    elements.previewState.textContent = previewState;
    const camera = state.camera || {};
    elements.cameraConfig.textContent =
      `#${camera.index ?? "—"} · ${camera.width ?? "—"}×${camera.height ?? "—"} · ${camera.fps ?? "—"} FPS · 校正 ${camera.rotation_degrees ?? 0}°`;
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
      STARTING: "正在打开 COM4 并释放全部六轴扭矩…",
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
    if (state.neutral_ready) {
      elements.neutralProgressText.textContent = "已完成 · 可以 ARM";
    } else if (state.neutral_collecting) {
      const perChannel = state.neutral_channel_progress || {};
      const detail = Object.entries(channelLabels)
        .map(([name, label]) => `${label} ${Number(perChannel[name]) || 0} 帧`)
        .join(" · ");
      elements.neutralProgressText.textContent =
        `正在采集 · 剩余 ${Math.ceil(neutralRemaining)} 秒 · ${detail}`;
    } else {
      elements.neutralProgressText.textContent =
        state.neutral_error ? `采集未完成：${state.neutral_error}` : "未完成 · 将连续采集 30 秒";
    }

    elements.connectButton.disabled = busy || countdownActive || !operator || !state.can_connect;
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
    // STOP remains available for an in-flight ARM, but not after torque release has
    // been confirmed: a release already in progress cannot be cancelled safely.
    elements.stopButton.disabled =
      !operator || releaseInProgress || (!state.armed && !armPending);
    elements.disconnectButton.disabled =
      busy || countdownActive || !operator || !state.can_disconnect;
    elements.connectButton.textContent = robotState === "CONNECTING" ? "连接中…" : "连接机器人";
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

    renderJoints(Array.isArray(state.joints) ? state.joints : []);
    elements.connectionState.textContent = `后台在线 · 客户端 ${clientId.slice(0, 8)}`;

    if (!videoFailed && previewState === "RUNNING") {
      elements.videoFallback.hidden = true;
      elements.videoFeed.hidden = false;
    }
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
      role.textContent = joint.role === "fixed" ? "固定保持" : "人体跟随";
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

  async function pollStatus() {
    if (polling) return;
    polling = true;
    try {
      const state = await api("/api/status");
      render(state);
    } catch (error) {
      cancelNeutralCountdown(false);
      cancelArmCountdown(false);
      elements.connectionState.textContent = "后台连接中断";
      elements.globalMessage.textContent = `无法读取后台状态：${error.message}`;
      elements.globalMessage.classList.add("is-error");
      disableAllControls();
    } finally {
      polling = false;
    }
  }

  async function runCommand(path, body = null) {
    if (busy) return;
    busy = true;
    pendingCommand = path;
    let commandError = null;
    disableAllControls();
    try {
      const options = { method: "POST" };
      if (body !== null) options.body = JSON.stringify(body);
      const result = await api(path, options);
      elements.globalMessage.textContent = result.message || "命令已执行";
      await pollStatus();
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
  }

  async function stopFollowing() {
    if (!currentState?.operator) return;
    elements.stopButton.disabled = true;
    elements.globalMessage.textContent = "正在停止跟随…";
    try {
      const result = await api("/api/disarm", { method: "POST" });
      await pollStatus();
      elements.globalMessage.textContent = result.message || "已停止跟随；本次命令未请求释放扭矩";
      elements.globalMessage.classList.remove("is-error", "is-armed");
    } catch (error) {
      elements.globalMessage.textContent = error.message;
      elements.globalMessage.classList.add("is-error");
    }
  }

  function disableAllControls() {
    [
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
    ].forEach((button) => { button.disabled = true; });
  }

  function showNeutralCountdownMessage() {
    elements.globalMessage.textContent =
      `中立姿势将在 ${neutralCountdownRemaining} 秒后开始连续采集 30 秒，请走到摄像头前并保持自然姿势。再次点击按钮可取消。`;
    elements.globalMessage.classList.remove("is-error", "is-armed");
  }

  function updateNeutralCountdownDisplay() {
    elements.neutralButton.disabled = false;
    elements.neutralButton.textContent = `取消倒计时 · ${neutralCountdownRemaining} 秒`;
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

    neutralCountdownRemaining = ACTION_COUNTDOWN_SECONDS;
    neutralCountdownTimer = window.setInterval(() => {
      neutralCountdownRemaining -= 1;
      if (neutralCountdownRemaining > 0) {
        updateNeutralCountdownDisplay();
        return;
      }
      window.clearInterval(neutralCountdownTimer);
      neutralCountdownTimer = null;
      neutralCountdownRemaining = 0;
      if (!neutralCountdownAllowed(currentState)) {
        if (currentState) render(currentState);
        elements.globalMessage.textContent = "状态已变化，中立采集已取消。";
        elements.globalMessage.classList.add("is-error");
        return;
      }
      elements.globalMessage.textContent = "正在开始 30 秒中立姿势采集…";
      runCommand("/api/neutral");
    }, 1000);
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
    updateArmCountdownDisplay();
  }

  elements.baseCalibrationStartButton.addEventListener("click", () => {
    runCommand("/api/base-calibration/start", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
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
    runCommand("/api/connect", { confirmation: CONTROL_CONFIRMATION_CODE });
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

  elements.disconnectButton.addEventListener("click", () => {
    runCommand("/api/disconnect", {
      confirmation: CONTROL_CONFIRMATION_CODE,
    });
  });

  elements.videoFeed.addEventListener("load", () => {
    videoFailed = false;
    elements.videoFallback.hidden = true;
    elements.videoFeed.hidden = false;
  });

  elements.videoFeed.addEventListener("error", () => {
    videoFailed = true;
    elements.videoFeed.hidden = true;
    elements.videoFallback.hidden = false;
  });

  function disarmOnLeave() {
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
      headers: { "X-Client-Id": clientId },
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
