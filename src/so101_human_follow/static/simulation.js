(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const dialog = byId("simulationDialog");
  if (!dialog) return;
  const elements = {
    message: byId("simulationMessage"), image: byId("simulationFrame"), hint: byId("simulationFrameHint"),
    joints: byId("simulationJointControls"), progress: byId("simulationProgress"), contacts: byId("simulationContacts"),
    frames: byId("simulationKeyframes"), capture: byId("simulationCaptureButton"),
    move: byId("simulationMoveTime"), hold: byId("simulationHoldTime"), name: byId("simulationActionName"),
    description: byId("simulationActionDescription"),
    stepTime: byId("simulationStepTime"), actionTime: byId("simulationActionTime"),
    captureMode: byId("simulationCaptureMode"),
    cancelEdit: byId("simulationCancelEditButton"),
    saved: byId("simulationSavedActions"), execute: byId("simulationExecuteButton"),
    save: byId("simulationSaveButton"), load: byId("simulationLoadButton"), remove: byId("simulationDeleteButton"),
    recordings: byId("simulationSavedRecordings"), recordingDescription: byId("simulationRecordingDescription"),
    replay: byId("simulationPreviewRecordingButton"),
    previewFollower: byId("simulationPreviewFollowerButton"), executeFollower: byId("simulationExecuteFollowerButton"),
    followerRecordingNote: byId("simulationFollowerRecordingNote"),
    actionKind: byId("simulationActionKind"), compositionEditor: byId("simulationCompositionEditor"),
    compositionSource: byId("simulationCompositionSource"), compositionAdd: byId("simulationCompositionAdd"),
    compositionItems: byId("simulationCompositionItems"), keyframeHelp: byId("simulationKeyframeHelp"),
    returnStart: byId("simulationReturnToStart"), hardware: byId("simulationHardwareState"),
    trialStatus: byId("simulationTrialStatus"),
  };
  let simulation = null;
  let targets = {};
  let frames = [];
  let compositionItems = [];
  let savedActions = new Map();
  let selected = -1;
  let savedId = null;
  let dirty = false;
  let hardwareConnected = false;
  let trialStarting = false;
  let trial = null;
  let timer = null;
  let poseTimer = null;
  let cameraTimer = null;
  let controller = null;
  let objectUrl = null;
  let generation = 0;
  let hardwarePollAt = 0;
  let poseRevision = 0;
  let poseSending = false;
  let draftOrder = [];
  let pendingRecording = null;
  let pendingRecordingId = null;
  let actionRecording = null;
  let actionObservedLimits = {};
  let recordings = new Map();
  let replayBlocked = false;
  let replayPending = false;
  let libraryBusy = false;
  const deletedActions = new Set();
  const MAX_COMPOSITION_ITEMS = 12;
  const jointElements = new Map();
  const DEFAULT_JOINT_MOVE = 0.8;
  const START_APPROACH_S = 0.5;
  const CURRENT_RECORDING = "current";
  const teaching = window.so101CreateTeachingPanel({
    api, message,
    getRecording: () => pendingRecording ?? actionRecording,
    getRecordingPayload: () => {
      const id = pendingRecording ? pendingRecordingId : savedId;
      return id ? { recording_id: id } : { recording: pendingRecording ?? actionRecording };
    },
    setRecording: rememberRecording,
    getObservedLimits: () => pendingRecording?.ranges ?? actionObservedLimits,
    getSimulation: () => simulation,
    applyDraft,
    setReplayBlocked: blocked => { replayBlocked = blocked; updateReplay(); },
  });

  function headers() {
    const id = window.so101ControlClientId;
    if (typeof id !== "string" || !/^[A-Za-z0-9_-]{16,96}$/.test(id)) {
      throw new Error("网页控制标识未初始化，请刷新页面后重试。");
    }
    return { "Content-Type": "application/json", "X-Client-Id": id };
  }

  async function api(path, payload = null, signal = controller?.signal) {
    const response = await fetch(path, {
      method: payload === null ? "GET" : "POST", credentials: "same-origin", cache: "no-store",
      headers: headers(), signal, ...(payload === null ? {} : { body: JSON.stringify(payload) }),
    });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(value.error || value.message || `请求失败：${response.status}`), { status: response.status });
    return value;
  }

  function message(text, error = false) {
    elements.message.textContent = text;
    elements.message.classList.toggle("is-error", error);
  }

  function changed() {
    dirty = true;
    updateExecute();
  }

  function editingComposition() {
    return elements.actionKind.value === "composition";
  }

  function updateExecute() {
    const unavailable = !hardwareConnected || libraryBusy || trialStarting || Boolean(trial) || replayBlocked;
    const hasSteps = editingComposition() ? compositionItems.length : frames.length;
    elements.execute.disabled = unavailable || !savedId || dirty || !hasSteps;
    const selectedRecording = recordings.get(elements.recordings.value);
    elements.executeFollower.disabled = unavailable || !selectedRecording?.has_follower_calibration;
  }

  function showTrialStatus(text, error = false) {
    elements.trialStatus.textContent = text;
    elements.trialStatus.classList.toggle("is-error", error);
    elements.trialStatus.hidden = !text;
  }

  function setTargets(values) {
    targets = { ...targets, ...values };
    for (const [name, row] of jointElements) {
      if (!(name in targets)) continue;
      row.slider.value = targets[name];
      row.number.value = String(targets[name]);
    }
  }

  function editTarget(joint, value) {
    const numeric = Number(value);
    const { min, max } = jointElements.get(joint.name).number;
    if (!Number.isFinite(numeric) || numeric < Number(min) || numeric > Number(max)) {
      message(`ID${joint.id} 目标必须在 ${min}～${max} 之间。`, true);
      return;
    }
    setTargets({ [joint.name]: numeric });
    changed();
    poseRevision += 1;
    window.clearTimeout(poseTimer);
    poseTimer = window.setTimeout(sendPose, 100);
  }

  async function sendPose() {
    if (poseSending || !dialog.open || !simulation) return;
    poseSending = true;
    const revision = poseRevision;
    try {
      const unlocked = {};
      for (const [name, row] of jointElements) if (!row.lock.checked) unlocked[name] = targets[name];
      if (Object.keys(unlocked).length) {
        const result = await api("/api/simulation/pose", { targets: unlocked });
        renderState(result.simulation);
      }
    } catch (error) { if (error.name !== "AbortError") message(error.message, true); }
    finally {
      poseSending = false;
      if (revision !== poseRevision && dialog.open) void sendPose();
    }
  }

  function refreshDraftOrder() {
    const active = [...jointElements].filter(([, row]) => !row.lock.checked).map(([joint]) => joint);
    draftOrder = orderedJoints(Object.fromEntries(active.map(joint => [joint, true])), draftOrder);
    const sequential = elements.captureMode.value === "sequential";
    for (const [joint, row] of jointElements) {
      const rank = draftOrder.indexOf(joint) + 1;
      const ranks = draftOrder.map((_, index) => new Option(String(index + 1), String(index + 1)));
      const options = rank === 0 ? [new Option("固定", "")] : sequential ? ranks : [new Option("同时", ""), ...ranks];
      row.order.replaceChildren(...options);
      row.order.value = sequential && rank ? String(rank) : "";
      row.order.disabled = row.start.disabled = row.slider.disabled = row.number.disabled = row.lock.checked;
      row.move.disabled = row.hold.disabled = row.lock.checked;
      row.hardware.disabled = row.lock.checked || !hardwareConnected;
      renderRecordedRange(joint, row);
    }
    renderDraftTiming();
  }

  function finishEditingStep() {
    selected = -1;
    for (const row of jointElements.values()) row.start.value = "";
    renderDraftTiming();
    renderFrames();
  }

  function seconds(value) {
    return Number.isFinite(value) ? `${value.toFixed(2)} 秒` : "—";
  }

  function timingTotals(mode, order, timing, starts) {
    let moveSum = 0;
    let holdSum = 0;
    let longest = 0;
    let explicitStarts = 0;
    for (const joint of order) {
      const { duration_s: move, hold_s: hold } = timing[joint];
      moveSum += move;
      holdSum += hold;
      longest = Math.max(longest, move + hold);
      if (Object.hasOwn(starts, joint)) explicitStarts += 1;
    }
    const sequential = mode === "sequential";
    const approach = START_APPROACH_S * (sequential ? explicitStarts : Math.min(explicitStarts, 1));
    return { moveSum, holdSum, approach, elapsed: approach + (sequential ? moveSum + holdSum : longest) };
  }

  function renderDraftTiming() {
    const timing = {};
    const starts = {};
    for (const joint of draftOrder) {
      const row = jointElements.get(joint);
      const move = row.move.validity.valid ? row.move.valueAsNumber : NaN;
      const hold = row.hold.validity.valid ? row.hold.valueAsNumber : NaN;
      timing[joint] = { duration_s: move, hold_s: hold };
      if (row.start.value !== "") starts[joint] = row.start.valueAsNumber;
    }
    const totals = timingTotals(elements.captureMode.value, draftOrder, timing, starts);
    elements.move.value = seconds(totals.moveSum);
    elements.hold.value = seconds(totals.holdSum);
    elements.stepTime.value = seconds(totals.elapsed);
  }

  function renderRecordedRange(joint, row) {
    const range = actionObservedLimits[joint];
    row.recorded.hidden = !range;
    if (!range) return;
    const [low, high] = range;
    const inLimits = low >= Number(row.number.min) && high <= Number(row.number.max);
    row.recordedLabel.textContent = `录制实际 ${low.toFixed(2)}～${high.toFixed(2)}${inLimits ? "" : " · 超出当前编排范围"}`;
    for (const button of row.recordedButtons) button.disabled = row.lock.checked || !inLimits;
  }

  function createJoint(joint) {
    const section = document.createElement("div"); section.className = "simulation-joint";
    const heading = document.createElement("div"); heading.className = "simulation-joint-header";
    const title = document.createElement("strong"); title.textContent = `ID${joint.id} · ${joint.label}`;
    const lockLabel = document.createElement("label");
    const lock = document.createElement("input"); lock.type = "checkbox";
    lockLabel.append(lock, document.createTextNode(" 固定")); heading.append(title, lockLabel);
    const endpoints = document.createElement("div"); endpoints.className = "simulation-joint-endpoints";
    const start = document.createElement("input"); start.type = "number"; start.step = "any";
    start.placeholder = "沿用位置";
    start.setAttribute("aria-label", `ID${joint.id} 从 起点位置`);
    const number = document.createElement("input"); number.type = "number"; number.step = "any";
    number.setAttribute("aria-label", `ID${joint.id} 到 终点位置`);
    const order = document.createElement("select");
    order.setAttribute("aria-label", `ID${joint.id} 执行顺序`);
    const move = document.createElement("input");
    const hold = document.createElement("input");
    for (const [field, label, minimum, initial] of [
      [move, "运动时间", 0.1, DEFAULT_JOINT_MOVE],
      [hold, "停留时间", 0, 0],
    ]) {
      field.type = "number";
      field.step = "any";
      field.min = minimum;
      field.max = "10";
      field.required = true;
      field.value = initial;
      field.setAttribute("aria-label", `ID${joint.id} ${label}（秒）`);
      field.addEventListener("input", renderDraftTiming);
    }
    for (const [caption, field] of [["从（起点）", start], ["到（终点）", number], ["顺序", order], ["运动时间（秒）", move], ["停留时间（秒）", hold]]) {
      const label = document.createElement("label");
      label.append(document.createTextNode(caption), field);
      endpoints.append(label);
    }
    const edit = document.createElement("div"); edit.className = "simulation-joint-edit";
    const slider = document.createElement("input"); slider.type = "range"; slider.step = "0.1"; slider.min = joint.min; slider.max = joint.max;
    slider.setAttribute("aria-label", `ID${joint.id} 仿真终点滑块`);
    const hardware = document.createElement("button"); hardware.type = "button"; hardware.className = "button button-danger-outline";
    hardware.textContent = "真机移动"; hardware.disabled = !hardwareConnected;
    hardware.title = "只在点击后控制这一个真机关节，滑块本身不控制真机";
    const readout = document.createElement("small"); readout.className = "simulation-joint-readout";
    const recorded = document.createElement("div"); recorded.className = "simulation-recorded-range"; recorded.hidden = true;
    const recordedLabel = document.createElement("small");
    const recordedButtons = [
      stepButton("最小 → 最大", () => useRecordedRange(false)),
      stepButton("最大 → 最小", () => useRecordedRange(true)),
    ];
    function useRecordedRange(reverse) {
      const range = actionObservedLimits[joint.name];
      if (!range) return;
      start.value = range[reverse ? 1 : 0];
      editTarget(joint, range[reverse ? 0 : 1]);
      renderDraftTiming();
    }
    recorded.append(recordedLabel, ...recordedButtons);
    edit.append(slider, hardware); section.append(heading, endpoints, edit, readout, recorded); elements.joints.append(section);
    jointElements.set(joint.name, { slider, start, number, order, move, hold, lock, hardware, readout, recorded, recordedLabel, recordedButtons });
    slider.addEventListener("input", () => editTarget(joint, slider.valueAsNumber));
    number.addEventListener("change", () => editTarget(joint, number.valueAsNumber));
    start.addEventListener("input", renderDraftTiming);
    order.addEventListener("change", () => {
      if (!order.value) return;
      const current = draftOrder.indexOf(joint.name);
      draftOrder.splice(current, 1);
      draftOrder.splice(Number(order.value) - 1, 0, joint.name);
      elements.captureMode.value = "sequential";
      refreshDraftOrder();
    });
    lock.addEventListener("change", () => {
      refreshDraftOrder();
      changed();
    });
    hardware.addEventListener("click", async () => {
      if (!number.validity.valid || !Number.isFinite(number.valueAsNumber)) {
        return message(`请填写 ID${joint.id} 的有效终点位置。`, true);
      }
      hardware.disabled = true;
      const sequence = window.so101InvalidateInteraction?.();
      try {
        await api("/api/joints/move", { joint: joint.name, position: number.valueAsNumber, return_to_start: false,
          ...(Number.isInteger(sequence) ? { client_sequence: sequence } : {}) });
        message(`已提交 ID${joint.id} 真机移动，请观察实际运动。`);
      } catch (error) { message(error.message, true); }
      finally { hardware.disabled = lock.checked || !hardwareConnected; }
    });
  }

  function renderState(state, resetTargets = false) {
    simulation = state;
    let addedJoint = false;
    for (const joint of state.joints) {
      if (!jointElements.has(joint.name)) {
        createJoint(joint);
        addedJoint = true;
      }
      const row = jointElements.get(joint.name);
      row.readout.textContent = `编排允许 ${joint.min.toFixed(1)}～${joint.max.toFixed(1)} · 仿真实测 ${joint.position.toFixed(2)} · 模型角度 ${joint.angle_deg.toFixed(1)}°`;
      row.slider.min = row.start.min = row.number.min = joint.min;
      row.slider.max = row.start.max = row.number.max = joint.max;
      renderRecordedRange(joint.name, row);
      if (resetTargets || !(joint.name in targets)) setTargets({ [joint.name]: joint.target });
    }
    if (addedJoint) refreshDraftOrder();
    elements.progress.textContent = `${state.elapsed_s.toFixed(1)} / ${state.duration_s.toFixed(1)} 秒${state.playing ? " · 预览中" : ""}`;
    elements.contacts.textContent = `接触点：${state.contacts}`;
    teaching.render();
    if (state.error) message(state.error, true);
  }

  async function readHardware() {
    const requestedTrial = trial;
    const value = await api("/api/joints");
    hardwareConnected = Boolean(value.connected ?? value.robot_connected);
    elements.hardware.textContent = hardwareConnected ? "真机已连接。点击单轴“真机移动”可分别测试，预览与拖动滑块不控制真机。" : "真机未连接，仍可独立使用仿真与动作编排。";
    for (const row of jointElements.values()) row.hardware.disabled = row.lock.checked || !hardwareConnected;
    if (requestedTrial && requestedTrial === trial && value.motion) {
      const motion = value.motion;
      if (motion.id === trial.id) {
        showTrialStatus(`真机试运行 · 第 ${motion.step}/${trial.steps} 步`);
      } else {
        if (motion.id !== null) showTrialStatus("试运行已被其他动作中断。", true);
        else if (motion.state === "COMPLETE") showTrialStatus("真机试运行完成。");
        else showTrialStatus(motion.error || "真机试运行已停止。", true);
        trial = null;
      }
    }
    updateExecute();
  }

  async function refreshFrame(signal) {
    const response = await fetch("/api/simulation/frame", { headers: headers(), credentials: "same-origin", cache: "no-store", signal });
    if (!response.ok) {
      const value = await response.json().catch(() => ({}));
      throw new Error(value.error || "仿真画面读取失败。");
    }
    const blob = await response.blob();
    if (signal.aborted || !dialog.open) return;
    const previous = objectUrl;
    objectUrl = URL.createObjectURL(blob);
    elements.image.src = objectUrl; elements.image.classList.add("has-frame"); elements.hint.hidden = true;
    if (previous) URL.revokeObjectURL(previous);
  }

  async function tick(currentGeneration) {
    if (!dialog.open || currentGeneration !== generation || !controller) return;
    if (Date.now() - hardwarePollAt > (trial ? 500 : 1500)) {
      hardwarePollAt = Date.now();
      void readHardware().catch(error => {
        if (error.name !== "AbortError" && trial && currentGeneration === generation) {
          showTrialStatus(`暂时无法读取真机进度：${error.message}`, true);
        }
      });
    }
    try {
      const result = await api("/api/simulation/status");
      if (currentGeneration !== generation) return;
      renderState(result.simulation);
      await refreshFrame(controller.signal);
    } catch (error) { if (error.name !== "AbortError") message(error.message, true); }
    finally {
      if (dialog.open && currentGeneration === generation) timer = window.setTimeout(() => tick(currentGeneration), 150);
    }
  }

  async function open() {
    if (dialog.open) return;
    dialog.showModal();
    const currentGeneration = ++generation;
    controller = new AbortController();
    simulation = null;
    teaching.start();
    message("正在加载本地 MuJoCo 模型和物理引擎…");
    elements.hint.hidden = false; elements.hint.textContent = "正在准备仿真画面…";
    try {
      const result = await api("/api/simulation/open", {});
      if (!dialog.open || currentGeneration !== generation) return;
      renderState(result.simulation, true);
      message(result.simulation.mapping_note);
      void refreshLibrary().catch(error => {
        if (error.name !== "AbortError" && currentGeneration === generation) message(error.message, true);
      });
      void tick(currentGeneration);
    } catch (error) {
      if (error.name !== "AbortError") { message(error.message, true); elements.hint.textContent = error.message; }
    }
  }

  function actionPayload(includeRecording = false) {
    const common = {
      ...(savedId ? { id: savedId } : {}),
      name: elements.name.value.trim() || "未命名预览", description: elements.description.value.trim(),
      return_to_start: elements.returnStart.checked,
    };
    if (editingComposition()) {
      if (!compositionItems.length) throw new Error("请先加入至少一个已保存动作。");
      return { ...common, kind: "composition", items: compositionItems };
    }
    if (!frames.length) throw new Error("请先加入至少一个姿势。");
    return {
      ...common, frames,
      ...(includeRecording && actionRecording ? { recording: actionRecording } : {}),
    };
  }

  function applyDraft(draft) {
    elements.actionKind.value = "keyframes";
    compositionItems = [];
    savedId = draft.id ?? null;
    elements.saved.value = savedId ?? "";
    elements.name.value = draft.name ?? "";
    elements.description.value = draft.description ?? "";
    elements.returnStart.checked = Boolean(draft.return_to_start);
    frames = loadFrames(draft.frames);
    actionRecording = draft.recording;
    actionObservedLimits = draft.observed_limits ?? {};
    pendingRecording = null;
    pendingRecordingId = null;
    renderRecordingOptions();
    changed();
    finishEditingStep();
    renderComposition();
    if (frames.length) editFrame(0);
    const note = draft.limited_joints?.length ? " 部分录制值超出当前编排范围，生成的步骤已按当前范围限制。" : "";
    const source = draft.source === "follower" ? "按从臂实测轨迹" : "按主臂录制轨迹";
    message(`${source}生成了 ${frames.length} 个动作步骤。可调整起止位置、执行顺序和时间后命名保存。${note}`);
  }

  function orderedJoints(targets, previous = []) {
    const retained = previous.filter(joint => Object.hasOwn(targets, joint));
    return [...retained, ...Object.keys(targets).filter(joint => !retained.includes(joint))];
  }

  function loadFrames(savedFrames) {
    return savedFrames.map(frame => ({
      ...frame,
      mode: frame.mode ?? "parallel",
      joint_order: orderedJoints(frame.targets, frame.joint_order),
    }));
  }

  function frameTiming(frame) {
    if (frame.joint_timing) return frame.joint_timing;
    const sequential = frame.mode === "sequential";
    const move = frame.duration_s ?? DEFAULT_JOINT_MOVE;
    const hold = frame.hold_s ?? 0;
    return Object.fromEntries(frame.joint_order.map((joint, index) => [
      joint,
      {
        duration_s: sequential ? Math.max(0.1, move / frame.joint_order.length) : move,
        hold_s: !sequential || index === frame.joint_order.length - 1 ? hold : 0,
      },
    ]));
  }

  function renderActionTime() {
    if (editingComposition()) {
      let elapsed = 0;
      let available = true;
      for (const item of compositionItems) {
        const source = savedActions.get(item.action_id);
        const length = item.playback === "follower_trajectory" ? source?.follower_duration_s : source?.duration_s;
        if (!source || !Number.isFinite(length)) { available = false; break; }
        elapsed += item.repeat * (length + item.hold_s + (item.playback === "follower_trajectory" ? 1 : 0));
      }
      const returns = compositionItems.length && elements.returnStart.checked;
      const total = compositionItems.length && available ? seconds(elapsed + (returns ? 1 : 0)) : "—";
      elements.actionTime.textContent = `整套动作预计耗时 ${total}${returns && available ? "（含回位 1 秒）" : ""}`;
      return;
    }
    const elapsed = frames.reduce((total, frame) => total + timingTotals(
      frame.mode, frame.joint_order, frameTiming(frame), frame.start_positions ?? {},
    ).elapsed, 0);
    const returns = frames.length && elements.returnStart.checked;
    const total = frames.length ? seconds(elapsed + (returns ? 1 : 0)) : "—";
    elements.actionTime.textContent = `整套动作预计耗时 ${total}${returns ? "（含回位 1 秒）" : ""}`;
  }

  function stepButton(label, callback, disabled = false, accessibleLabel = label) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.disabled = disabled;
    button.setAttribute("aria-label", accessibleLabel);
    button.addEventListener("click", callback);
    return button;
  }

  function editFrame(index) {
    selected = index;
    const frame = frames[index];
    setTargets(frame.targets);
    draftOrder = [...frame.joint_order];
    elements.captureMode.value = frame.mode;
    const timing = frameTiming(frame);
    for (const [joint, row] of jointElements) {
      row.lock.checked = !Object.hasOwn(frame.targets, joint);
      row.start.value = frame.start_positions?.[joint] ?? "";
      row.move.value = timing[joint]?.duration_s ?? DEFAULT_JOINT_MOVE;
      row.hold.value = timing[joint]?.hold_s ?? 0;
    }
    refreshDraftOrder();
    renderFrames();
    const jointSettings = elements.captureMode.closest("details");
    if (jointSettings) jointSettings.open = true;
    elements.captureMode.scrollIntoView({ block: "start" });
    poseRevision += 1;
    void sendPose();
  }

  function renderFrameSettings(item, frame, index) {
    const settings = document.createElement("div");
    settings.className = "simulation-step-settings";
    const modeLabel = document.createElement("label");
    modeLabel.textContent = "执行方式";
    const mode = document.createElement("select");
    mode.append(new Option("同时运动", "parallel"), new Option("依次运动", "sequential"));
    mode.value = frame.mode;
    modeLabel.append(mode);
    settings.append(modeLabel);

    const jointList = document.createElement("ol");
    jointList.className = "simulation-joint-order";
    jointList.setAttribute("aria-label", `步骤 ${index + 1} 的关节顺序`);
    const detail = document.createElement("p");
    detail.className = "simulation-step-detail";
    function jointLabel(joint) {
      const servo = simulation?.joints.find(item => item.name === joint);
      return servo ? `ID${servo.id} · ${servo.label}` : joint;
    }
    function renderJointOrder() {
      const sequential = frame.mode === "sequential";
      const timing = frameTiming(frame);
      const totals = timingTotals(frame.mode, frame.joint_order, timing, frame.start_positions ?? {});
      const approach = totals.approach ? `（含就位 ${seconds(totals.approach)}）` : "";
      detail.textContent = `本步预计耗时 ${seconds(totals.elapsed)}${approach}`;
      jointList.replaceChildren();
      frame.joint_order.forEach((joint, position) => {
        const row = document.createElement("li");
        const label = document.createElement("span");
        const title = document.createElement("strong");
        title.textContent = `${sequential ? `${position + 1}. ` : ""}${jointLabel(joint)}`;
        const endpoints = document.createElement("small");
        endpoints.textContent = `${frame.start_positions?.[joint] ?? "沿用位置"} → ${frame.targets[joint]}`;
        const times = document.createElement("small");
        times.textContent = `运动 ${seconds(timing[joint].duration_s)} · 停留 ${seconds(timing[joint].hold_s)}`;
        label.append(title, endpoints, times);
        row.append(label);
        function moveJoint(offset) {
          frame.joint_timing = frameTiming(frame);
          const order = frame.joint_order;
          const target = position + offset;
          [order[position], order[target]] = [order[target], order[position]];
          if (selected === index) {
            draftOrder = [...order];
            refreshDraftOrder();
          }
          changed();
          renderJointOrder();
        }
        if (sequential) row.append(
          stepButton("提前", () => moveJoint(-1), position === 0, `提前 ${jointLabel(joint)}`),
          stepButton("延后", () => moveJoint(1), position === frame.joint_order.length - 1, `延后 ${jointLabel(joint)}`),
        );
        jointList.append(row);
      });
    }

    mode.addEventListener("change", () => {
      frame.joint_timing = frameTiming(frame);
      delete frame.duration_s;
      delete frame.hold_s;
      frame.mode = mode.value;
      if (selected === index) {
        elements.captureMode.value = frame.mode;
        refreshDraftOrder();
      }
      changed();
      renderJointOrder();
      renderActionTime();
    });
    item.append(settings, detail, jointList);
    renderJointOrder();
  }

  function renderFrames() {
    elements.frames.replaceChildren();
    renderActionTime();
    const editing = frames[selected];
    elements.capture.textContent = editing ? `更新步骤 ${selected + 1}` : "添加动作步骤";
    if (elements.cancelEdit) elements.cancelEdit.hidden = !editing;
    if (!frames.length) {
      const empty = document.createElement("li"); empty.className = "simulation-empty"; empty.textContent = "设好关节起点、终点和顺序后，点击“添加动作步骤”。"; elements.frames.append(empty); return;
    }
    frames.forEach((frame, index) => {
      const item = document.createElement("li"); item.className = "simulation-keyframe"; item.classList.toggle("is-selected", index === selected);
      const header = document.createElement("div");
      header.className = "simulation-step-header";
      const title = document.createElement("strong");
      title.textContent = `步骤 ${index + 1} · ${frame.joint_order.length} 轴`;
      const controls = document.createElement("div");
      controls.className = "simulation-step-controls";
      function moveStep(offset) {
        const target = index + offset;
        [frames[index], frames[target]] = [frames[target], frames[index]];
        selected = frames.indexOf(editing);
        changed();
        renderFrames();
      }
      controls.append(
        stepButton("修改关节和时间", () => editFrame(index), false, `修改步骤 ${index + 1} 的关节和时间`),
        stepButton("前移", () => moveStep(-1), index === 0, `前移步骤 ${index + 1}`),
        stepButton("后移", () => moveStep(1), index === frames.length - 1, `后移步骤 ${index + 1}`),
        stepButton("移除", () => {
          frames.splice(index, 1);
          selected = frames.indexOf(editing);
          changed();
          if (editing && selected === -1) finishEditingStep();
          else renderFrames();
        }, false, `移除步骤 ${index + 1}`),
      );
      header.append(title, controls);
      item.append(header);
      renderFrameSettings(item, frame, index);
      elements.frames.append(item);
    });
  }

  async function refreshSaved(selectId = savedId) {
    const result = await api("/api/simulation/actions");
    savedActions = new Map(result.actions.filter(action => !deletedActions.has(action.id)).map(action => [action.id, action]));
    elements.saved.replaceChildren(new Option("选择已保存的动作", ""));
    for (const action of savedActions.values()) {
      const detail = action.kind === "composition" ? `${action.item_count} 项组合${action.compatible ? "" : "（需复核）"}`
        : action.recording_only ? "原始录制" : `${action.frame_count} 姿势${action.compatible ? "" : "（需复核）"}`;
      const option = new Option(`${action.name} · ${detail}`, action.id);
      option.dataset.name = action.name;
      elements.saved.add(option);
    }
    elements.saved.value = selectId && [...elements.saved.options].some(option => option.value === selectId) ? selectId : "";
    updateLibraryControls();
    renderComposition();
  }

  function updateLibraryControls() {
    elements.saved.disabled = libraryBusy;
    elements.save.disabled = libraryBusy;
    elements.load.disabled = elements.remove.disabled = libraryBusy || !elements.saved.value;
    elements.actionKind.disabled = libraryBusy || Boolean(savedId);
    elements.compositionSource.disabled = libraryBusy || !editingComposition();
    elements.compositionAdd.disabled = libraryBusy || !elements.compositionSource.value || compositionItems.length >= MAX_COMPOSITION_ITEMS;
  }

  function renderComposition() {
    const active = editingComposition();
    elements.frames.hidden = elements.keyframeHelp.hidden = active;
    elements.compositionEditor.hidden = !active;
    elements.capture.disabled = active;
    if (!active) {
      renderActionTime();
      return;
    }
    const sourcePicker = elements.compositionSource;
    const selectedSource = sourcePicker.value;
    const sources = [...savedActions.values()].filter(action =>
      action.kind !== "composition" && action.id !== savedId && (action.frame_count || Number.isFinite(action.follower_duration_s)),
    );
    sourcePicker.replaceChildren(new Option("选择已保存动作", ""), ...sources.map(action => new Option(action.name, action.id)));
    sourcePicker.value = sources.some(action => action.id === selectedSource) ? selectedSource : "";
    elements.compositionItems.replaceChildren();
    compositionItems.forEach((item, index) => {
      const source = savedActions.get(item.action_id);
      const row = document.createElement("li");
      row.className = "simulation-keyframe";
      const header = document.createElement("div");
      header.className = "simulation-step-header";
      const title = document.createElement("strong");
      title.textContent = `${index + 1}. ${source?.name ?? "缺少已保存动作"}`;
      const buttons = document.createElement("div");
      buttons.className = "simulation-step-controls";
      function move(offset) {
        const destination = index + offset;
        [compositionItems[index], compositionItems[destination]] = [compositionItems[destination], compositionItems[index]];
        changed();
        renderComposition();
      }
      buttons.append(
        stepButton("前移", () => move(-1), index === 0, `前移第 ${index + 1} 项动作`),
        stepButton("后移", () => move(1), index === compositionItems.length - 1, `后移第 ${index + 1} 项动作`),
        stepButton("移除", () => {
          compositionItems.splice(index, 1);
          changed();
          renderComposition();
        }, false, `移除第 ${index + 1} 项动作`),
      );
      header.append(title, buttons);
      row.append(header);
      if (source) {
        const settings = document.createElement("div");
        settings.className = "simulation-step-settings";
        function addSetting(text, input) {
          const label = document.createElement("label");
          label.textContent = text;
          label.append(input);
          settings.append(label);
        }
        const playback = document.createElement("select");
        if (source.frame_count || item.playback === "keyframes") {
          playback.add(new Option("关节步骤", "keyframes", false, item.playback === "keyframes"));
        }
        if (Number.isFinite(source.follower_duration_s) || item.playback === "follower_trajectory") {
          playback.add(new Option("从臂实测轨迹", "follower_trajectory", false, item.playback === "follower_trajectory"));
        }
        playback.addEventListener("change", () => {
          item.playback = playback.value;
          changed();
          renderActionTime();
        });
        addSetting("播放方式", playback);
        function addNumber(key, text, min, max, step) {
          const field = document.createElement("input");
          Object.assign(field, {type: "number", min, max, step, value: item[key], required: true});
          field.addEventListener("change", () => {
            if (!field.validity.valid) field.value = item[key];
            item[key] = field.valueAsNumber;
            changed();
            renderActionTime();
          });
          addSetting(text, field);
        }
        addNumber("repeat", "重复次数", 1, 8, 1);
        addNumber("hold_s", "每次播放后停留（秒）", 0, 10, "any");
        row.append(settings);
      }
      elements.compositionItems.append(row);
    });
    updateLibraryControls();
    renderActionTime();
  }

  function setLibraryBusy(busy) {
    libraryBusy = busy;
    updateLibraryControls();
    updateExecute();
  }

  function updateReplay() {
    const id = elements.recordings.value;
    const entry = recordings.get(id);
    const recent = id === CURRENT_RECORDING ? pendingRecording ?? actionRecording : null;
    const unavailable = !dialog.open || !simulation || replayBlocked || replayPending;
    elements.replay.disabled = unavailable || !(entry || recent);
    const hasFollower = Number.isFinite(entry?.follower_duration_s) || Boolean(recent?.follower_samples?.length);
    elements.previewFollower.disabled = unavailable || !hasFollower;
    elements.followerRecordingNote.hidden = !hasFollower;
    elements.followerRecordingNote.textContent = entry?.has_follower_calibration
      ? "从臂实测轨迹已保存，连接同一校准的从臂后可真机回放。"
      : "此轨迹可仿真预览。真机连续回放需要带从臂校准记录的新录制。";
    updateExecute();
    const description = entry?.description ?? recent?.description ?? "";
    if (elements.recordingDescription.textContent !== description) elements.recordingDescription.textContent = description;
    elements.recordingDescription.hidden = !description;
  }

  function renderRecordingOptions(selectedId = elements.recordings.value) {
    const options = Array.from(recordings.values(), entry =>
      new Option(`${entry.name} · ${entry.duration_s.toFixed(1)} 秒`, entry.id));
    if (pendingRecording ? !pendingRecordingId : actionRecording && !savedId) {
      options.push(new Option("本次录制", CURRENT_RECORDING));
    }
    const placeholder = options.length ? "选择已保存的原始录制" : "暂无原始录制";
    elements.recordings.replaceChildren(new Option(placeholder, ""), ...options);
    elements.recordings.value = options.some(option => option.value === selectedId) ? selectedId : "";
    elements.recordings.disabled = !options.length;
    updateReplay();
  }

  async function refreshRecordings() {
    const result = await api("/api/simulation/recordings");
    recordings = new Map(result.recordings.filter(entry => !deletedActions.has(entry.id)).map(entry => [entry.id, entry]));
    renderRecordingOptions();
  }

  function refreshLibrary() {
    return Promise.all([refreshSaved(), refreshRecordings()]);
  }

  function rememberRecording(recording, action) {
    pendingRecording = action ? { ...recording, name: action.name, description: action.description } : recording;
    pendingRecordingId = action?.id ?? null;
    if (action) {
      recordings.set(action.id, {
        id: action.id, name: action.name, description: action.description, duration_s: recording.duration_s,
        follower_duration_s: recording.follower_samples?.at(-1)?.time_s,
        has_follower_calibration: Boolean(recording.follower_calibration_fingerprint),
      });
      void refreshSaved(action.id).catch(error => {
        if (error.name !== "AbortError") message(`录制已保存。刷新动作列表失败：${error.message}`, true);
      });
    }
    renderRecordingOptions(pendingRecordingId ?? CURRENT_RECORDING);
  }

  byId("openSimulationButton").addEventListener("click", () => void open());
  async function closeSimulation() {
    try {
      if (await teaching.prepareClose()) dialog.close();
    } catch (error) {
      message(error.message, true);
    }
  }
  byId("closeSimulationButton").addEventListener("click", () => void closeSimulation());
  dialog.addEventListener("cancel", event => { event.preventDefault(); void closeSimulation(); });
  dialog.addEventListener("close", () => {
    teaching.close();
    ++generation; window.clearTimeout(timer); window.clearTimeout(poseTimer); window.clearTimeout(cameraTimer);
    controller?.abort(); controller = null;
    if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null;
    void api("/api/simulation/pause", {}, null).catch(() => {});
  });
  byId("simulationPreviewButton").addEventListener("click", async () => {
    try { const result = await api("/api/simulation/play", actionPayload()); renderState(result.simulation); message("正在进行 MuJoCo 物理仿真预览，未驱动真机。"); }
    catch (error) { message(error.message, true); }
  });
  elements.recordings.addEventListener("change", updateReplay);
  async function replayRecording(track, button) {
    if (button.disabled) return;
    const id = elements.recordings.value;
    const payload = id === CURRENT_RECORDING ? { recording: pendingRecording ?? actionRecording, track } : { recording_id: id, track };
    replayPending = true;
    updateReplay();
    try {
      const result = await api("/api/simulation/play_recording", payload);
      renderState(result.simulation);
      message(track === "follower" ? "正在预览从臂实测轨迹。" : "正在按原始采样时间回放主臂录制轨迹。");
    } catch (error) {
      if (error.name !== "AbortError") message(error.message, true);
    } finally {
      replayPending = false;
      updateReplay();
    }
  }
  elements.replay.addEventListener("click", () => replayRecording("leader", elements.replay));
  elements.previewFollower.addEventListener("click", () => replayRecording("follower", elements.previewFollower));
  byId("simulationPauseButton").addEventListener("click", async () => {
    try { renderState((await api("/api/simulation/pause", {})).simulation); message("仿真已暂停。"); } catch (error) { message(error.message, true); }
  });
  byId("simulationStopHardwareButton").addEventListener("click", async () => {
    const sequence = window.so101InvalidateInteraction?.();
    try { await api("/api/disarm", Number.isInteger(sequence) ? { client_sequence: sequence } : {}); message("已请求停止真机动作，停止后不会自动回位。"); }
    catch (error) { message(error.message, true); }
  });
  byId("simulationResetButton").addEventListener("click", async () => {
    try { renderState((await api("/api/simulation/reset", {})).simulation, true); finishEditingStep(); message("仿真姿势已重置，已保存动作不受影响。"); } catch (error) { message(error.message, true); }
  });
  elements.captureMode.addEventListener("change", refreshDraftOrder);
  elements.capture.addEventListener("click", () => {
    if (editingComposition()) return;
    if (!simulation) return message("请等待仿真初始化完成。", true);
    const updates = {};
    const starts = {};
    const timing = {};
    for (const [name, row] of jointElements) {
      if (row.lock.checked) continue;
      const value = row.number.valueAsNumber;
      if (!Number.isFinite(value) || value < Number(row.number.min) || value > Number(row.number.max)) {
        row.number.focus();
        return message("请把活动关节的终点填在显示范围内。", true);
      }
      updates[name] = value;
      if (!row.start.validity.valid) {
        row.start.focus();
        return message("请把活动关节的起点填在显示范围内，或留空沿用已有位置。", true);
      }
      if (row.start.value !== "") starts[name] = row.start.valueAsNumber;
      if (!row.move.validity.valid || !row.hold.validity.valid) {
        const invalid = row.move.validity.valid ? row.hold : row.move;
        invalid.focus();
        return message(`${invalid.getAttribute("aria-label")}须为 ${invalid.min}～${invalid.max} 秒。`, true);
      }
      timing[name] = { duration_s: row.move.valueAsNumber, hold_s: row.hold.valueAsNumber };
    }
    if (!Object.keys(updates).length) return message("请至少解除一个关节的固定。", true);
    const editing = frames[selected];
    const mode = elements.captureMode.value;
    const order = orderedJoints(updates, draftOrder);
    const frame = { targets: updates, joint_timing: timing, mode, joint_order: order,
      ...(Object.keys(starts).length ? { start_positions: starts } : {}) };
    if (selected >= 0) frames[selected] = frame;
    else { if (frames.length >= 32) return message("最多保存 32 个步骤。", true); frames.push(frame); }
    changed(); finishEditingStep();
    message(editing ? "步骤已更新。" : "已加入动作编排，可以继续添加下一步。");
  });
  elements.cancelEdit?.addEventListener("click", finishEditingStep);
  elements.returnStart.addEventListener("change", () => { changed(); renderActionTime(); });
  elements.name.addEventListener("input", changed);
  elements.description.addEventListener("input", changed);
  elements.actionKind.addEventListener("change", () => {
    changed();
    finishEditingStep();
    renderComposition();
  });
  elements.compositionSource.addEventListener("change", updateLibraryControls);
  elements.compositionAdd.addEventListener("click", () => {
    if (elements.compositionAdd.disabled) return;
    const source = savedActions.get(elements.compositionSource.value);
    if (!source) return;
    compositionItems.push({action_id: source.id, playback: source.frame_count ? "keyframes" : "follower_trajectory", repeat: 1, hold_s: 0});
    changed();
    renderComposition();
  });
  function clearActionEditor() {
    savedId = null; frames = []; actionRecording = null; actionObservedLimits = {}; dirty = false;
    compositionItems = [];
    elements.actionKind.value = "keyframes";
    elements.name.value = ""; elements.description.value = "";
    elements.returnStart.checked = true; elements.saved.value = "";
    renderRecordingOptions();
    finishEditingStep(); refreshDraftOrder(); teaching.render(); updateExecute();
    renderComposition();
  }

  byId("simulationNewButton").addEventListener("click", () => {
    if (libraryBusy) return;
    clearActionEditor();
    updateLibraryControls();
    message("已新建空动作，仿真当前姿势保留。");
  });
  elements.saved.addEventListener("change", updateLibraryControls);
  elements.save.addEventListener("click", async () => {
    if (libraryBusy) return;
    setLibraryBusy(true);
    try {
      if (!elements.name.value.trim()) {
        elements.name.focus();
        throw new Error("请填写动作名称。");
      }
      if (!elements.description.value.trim()) {
        elements.description.focus();
        throw new Error("请填写动作介绍。");
      }
      const result = await api("/api/simulation/actions", actionPayload(true));
      savedId = result.action.id; frames = loadFrames(result.action.frames); dirty = false;
      compositionItems = result.action.items ?? [];
      finishEditingStep(); updateExecute(); await refreshLibrary();
      message("动作已保存。Agent 可根据名称和介绍选择动作，也可点击“真机试运行此动作”。");
    } catch (error) { message(error.message, true); }
    finally { setLibraryBusy(false); }
  });
  elements.load.addEventListener("click", async () => {
    if (libraryBusy) return;
    setLibraryBusy(true);
    try {
      if (!elements.saved.value) throw new Error("请先选择已保存动作。");
      const { action, requires_review } = await api(`/api/simulation/actions/${encodeURIComponent(elements.saved.value)}`);
      savedId = action.id; frames = loadFrames(action.frames); dirty = Boolean(requires_review);
      elements.actionKind.value = action.kind === "composition" ? "composition" : "keyframes";
      compositionItems = action.items ?? [];
      pendingRecording = null;
      pendingRecordingId = null;
      actionRecording = action.recording ?? null;
      actionObservedLimits = action.observed_limits ?? {};
      elements.name.value = action.name; elements.description.value = action.description ?? "";
      elements.returnStart.checked = action.return_to_start;
      renderRecordingOptions(actionRecording && recordings.has(action.id) ? action.id : undefined);
      finishEditingStep(); refreshDraftOrder(); teaching.render(); updateExecute();
      renderComposition();
      if (editingComposition()) {
        message(requires_review ? action.review_error || "组合需复核并重新保存后再试运行。" : "组合已载入，可修改顺序、重复和停留时间。");
        return;
      }
      if (!frames.length && actionRecording) {
        message("原始录制已载入。打开「主从臂示教录制」并点击「生成动作草稿」即可编辑。");
        return;
      }
      message(requires_review ? "动作已载入。当前关节范围或校准已变化，请复核并重新保存后再试运行。" : "动作已载入，可在每个步骤里修改时间、执行方式和关节位置。");
    } catch (error) { message(error.message, true); }
    finally { setLibraryBusy(false); }
  });
  elements.remove.addEventListener("click", async () => {
    const id = elements.saved.value;
    if (!id || libraryBusy) return;
    const option = elements.saved.selectedOptions[0];
    const name = option.dataset.name || option.textContent;
    if (!window.confirm(`删除「${name}」？关联的原始录制（如有）也会删除，引用此动作的组合需要更新。`)) return;
    setLibraryBusy(true);
    try {
      await api(`/api/simulation/actions/${encodeURIComponent(id)}/delete`, {});
      deletedActions.add(id);
      savedActions.delete(id);
      [...elements.saved.options].find(item => item.value === id)?.remove();
      elements.saved.value = "";
      recordings.delete(id);
      if (pendingRecordingId === id) {
        pendingRecording = null;
        pendingRecordingId = null;
      }
      const keptDraft = savedId === id && dirty && (editingComposition() ? compositionItems.length : frames.length);
      if (savedId === id) {
        if (keptDraft) {
          savedId = null;
          actionRecording = null;
          actionObservedLimits = {};
          refreshDraftOrder();
          updateExecute();
        } else {
          clearActionEditor();
        }
      }
      renderRecordingOptions();
      renderComposition();
      teaching.render();
      message(`已删除「${name}」。${keptDraft ? "未保存的修改保留为新草稿。" : ""}`);
      void refreshSaved(elements.saved.value).catch(error => {
        if (error.name !== "AbortError") message(`动作已删除。刷新动作列表失败：${error.message}`, true);
      });
    } catch (error) { message(error.message, true); }
    finally { setLibraryBusy(false); }
  });
  async function executeTrial(id, payload) {
    trialStarting = true;
    updateExecute();
    showTrialStatus("正在启动真机试运行…");
    const sequence = window.so101InvalidateInteraction?.();
    try {
      const { agent_motion: motion } = await api(`/api/actions/${encodeURIComponent(id)}/execute`,
        Number.isInteger(sequence) ? { ...payload, client_sequence: sequence } : payload);
      trial = { id: motion.motion_id, steps: motion.step_count };
      showTrialStatus(`真机试运行 · 第 1/${trial.steps} 步`);
    } catch (error) { showTrialStatus(error.message, true); }
    finally { trialStarting = false; updateExecute(); }
  }
  elements.execute.addEventListener("click", () => {
    if (!elements.execute.disabled) void executeTrial(savedId, {});
  });
  elements.executeFollower.addEventListener("click", () => {
    if (elements.executeFollower.disabled) return;
    const id = elements.recordings.value;
    const entry = recordings.get(id);
    if (!window.confirm(`真机连续回放「${entry.name}」？从臂会先移动到录制起点，请确认运行空间畅通。`)) return;
    void executeTrial(id, { playback: "follower_trajectory" });
  });
  function cameraChanged() {
    window.clearTimeout(cameraTimer);
    cameraTimer = window.setTimeout(() => {
      void api("/api/simulation/camera", { azimuth: Number(byId("simulationAzimuth").value), elevation: -22, distance: Number(byId("simulationDistance").value) }).catch(error => message(error.message, true));
    }, 100);
  }
  byId("simulationAzimuth").addEventListener("input", cameraChanged);
  byId("simulationDistance").addEventListener("input", cameraChanged);
})();
