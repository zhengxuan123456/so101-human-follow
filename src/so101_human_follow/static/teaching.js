(() => {
  "use strict";

  const joints = [
    ["shoulder_pan.pos", "底座左右"], ["shoulder_lift.pos", "肩部抬落"],
    ["elbow_flex.pos", "肘部弯曲"], ["wrist_flex.pos", "腕部俯仰"],
    ["wrist_roll.pos", "腕部旋转"], ["gripper.pos", "夹爪开合"],
  ];

  window.so101CreateTeachingPanel = function ({
    api, getRecording, getRecordingPayload, setRecording, getObservedLimits, getSimulation, applyDraft,
    setReplayBlocked, message,
  }) {
    const byId = id => document.getElementById(id);
    const follower = window.so101FollowerControl;
    const leaderPort = byId("teachLeaderPort");
    const followerPort = byId("teachFollowerPort");
    const leaderStatus = byId("teachLeaderStatus");
    const followerStatus = byId("teachFollowerStatus");
    const followerCalibrationStatus = byId("teachFollowerCalibrationStatus");
    const mode = byId("teachRecordingMode");
    const name = byId("teachRecordingName");
    const description = byId("teachRecordingDescription");
    const syncCheck = byId("teachSyncCheck");
    const confirmed = byId("teachSyncConfirmed");
    const start = byId("teachStartRecording");
    const stop = byId("teachStopRecording");
    const extract = byId("teachExtractRecording");
    const recordStatus = byId("teachRecordingStatus");
    const leaderCommands = [...document.querySelectorAll("[data-teach-command]")];
    const followerCommands = [...document.querySelectorAll("[data-follower-command]")];
    const directions = new Map();
    const cells = new Map();
    let state = null;
    let pending = null;
    let lastError = "";
    let statusError = "";
    let seenResult = false;
    let followerConnected = false;
    let leaderPortEdited = false;
    let opened = false;
    let timer = null;
    let revision = 0;

    for (const [index, [joint, label]] of joints.entries()) {
      const caption = `ID${index + 1} ${label}`;
      const row = byId("teachJointRanges").insertRow();
      const heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = caption;
      row.append(heading);
      cells.set(joint, [row.insertCell(), row.insertCell(), row.insertCell(), row.insertCell()]);

      const field = document.createElement("label");
      field.textContent = caption;
      const select = document.createElement("select");
      select.append(new Option("同向", "1"), new Option("反向", "-1"));
      select.addEventListener("change", () => { confirmed.checked = false; render(); });
      field.append(select);
      byId("teachSyncDirections").append(field);
      directions.set(joint, select);
    }

    function rangeLabel(range, digits = 2) {
      if (!Array.isArray(range) || range.length !== 2 || !range.every(Number.isFinite)) return "—";
      return range.map(value => value.toFixed(digits)).join(" ～ ");
    }

    function render() {
      const shared = follower.state;
      const canControl = Boolean(shared?.operator && state?.robot_allowed);
      const canStart = canControl && !statusError;
      const busy = Boolean(pending);
      const leaderStage = state?.state;
      const followerCalibration = shared?.base_calibration;
      const recording = getRecording();
      const active = Boolean(state?.active);
      const recordingNow = Boolean(state?.recording);
      const sync = mode.value === "sync";
      const connected = Boolean(shared?.robot_connected);
      const calibrationPort = state?.port || state?.calibration_port || "";
      if (!leaderPortEdited && calibrationPort) leaderPort.value = calibrationPort;
      if (connected !== followerConnected) {
        followerConnected = connected;
        confirmed.checked = false;
      }
      if (followerPort.value !== follower.port) {
        followerPort.value = follower.port;
        confirmed.checked = false;
      }
      followerPort.disabled = follower.portDisabled || busy || active;
      leaderPort.disabled = busy || active;
      mode.disabled = busy || active;
      name.disabled = busy || active;
      description.disabled = busy || active;
      syncCheck.hidden = !sync;
      syncCheck.disabled = busy || active;
      const mainPort = leaderPort.value.trim();
      const otherPort = followerPort.value.trim();
      const samePort = Boolean(mainPort && mainPort.toUpperCase() === otherPort.toUpperCase());
      const calibrating = ["WAITING_MIDDLE", "MIDDLE_CAPTURED", "CALIBRATING_RANGE"].includes(leaderStage);
      const allowedStages = {
        "calibration/start": !active && Boolean(mainPort) && !(connected && samePort) && !followerCalibration?.active,
        "calibration/middle": leaderStage === "WAITING_MIDDLE",
        "calibration/range": leaderStage === "MIDDLE_CAPTURED",
        "calibration/save": leaderStage === "CALIBRATING_RANGE",
        "calibration/cancel": calibrating,
      };
      for (const button of leaderCommands) {
        const command = button.dataset.teachCommand;
        const allowed = command === "calibration/cancel" ? canControl : canStart;
        button.disabled = !allowed || busy || !allowedStages[command];
      }
      const followerFlags = {
        connect: shared?.can_connect && Boolean(otherPort),
        "calibration/start": shared?.can_start_base_calibration && Boolean(otherPort),
        "calibration/middle": shared?.can_capture_base_middle,
        "calibration/range": shared?.can_start_base_range,
        "calibration/save": shared?.can_finish_base_calibration,
        "calibration/cancel": shared?.can_cancel_base_calibration,
      };
      for (const button of followerCommands) {
        const command = button.dataset.followerCommand;
        button.disabled = !canStart || busy || active || !followerFlags[command];
        if (command === "connect") button.textContent = connected ? "从臂已连接" : "连接从臂";
      }
      const hasMetadata = Boolean(name.value.trim() && description.value.trim());
      start.disabled = !canStart || busy || active || !mainPort || !hasMetadata || (samePort && (sync || connected)) ||
        Boolean(followerCalibration?.active) || (sync && (!connected || !confirmed.checked || shared?.armed || shared?.agent_motion?.active));
      const resultReady = leaderStage === "RECORDED" && !seenResult;
      stop.disabled = !canControl || busy || (!recordingNow && !resultReady);
      stop.textContent = resultReady ? "取回已完成的录制" : "停止录制";
      extract.disabled = busy || active || !recording || !getSimulation();
      setReplayBlocked(busy || active);

      const leaderMessages = {
        WAITING_MIDDLE: "将主臂摆到六轴中位，再点击「确定六轴中位」。",
        MIDDLE_CAPTURED: "中位已确定。点击「采集六轴范围」后，逐轴移到两个行程端点。",
        CALIBRATING_RANGE: `正在采集六轴范围 · 剩余 ${Math.ceil(state?.calibration_remaining_s ?? 0)} 秒。逐轴移动完成后可以提前保存。`,
        CALIBRATED: "主臂校准已保存。",
      };
      leaderStatus.textContent = leaderMessages[leaderStage] || "";
      followerStatus.textContent = connected ? `已连接 · ${state?.follower_port || otherPort}`
        : state?.robot_allowed === false ? "当前服务未启用真机串口控制。" : "等待连接从臂。";
      followerCalibrationStatus.textContent = byId("baseCalibrationProgressText").textContent;

      const seconds = Number(state?.duration_s || 0).toFixed(1);
      const error = lastError || statusError || state?.error;
      recordStatus.classList.toggle("is-error", Boolean(error));
      let note = "填入主臂串口、动作名称和介绍，选择录制模式。";
      if (recordingNow) {
        note = `录制中 · ${seconds} 秒 · ${state.sample_count ?? 0} 帧${state.sync_active ? " · 从臂同步跟随" : ""}`;
      } else if (resultReady) {
        note = "录制已结束，点击「取回已完成的录制」。";
      } else if (samePort && (sync || connected)) {
        note = "主臂和从臂需要使用不同的串口。";
      } else if (recording) {
        const title = recording.name ? `「${recording.name}」 · ` : "";
        note = `已有录制 · ${title}${recording.samples.length} 帧，可回放原始录制或生成动作草稿。`;
      } else if (sync && !connected) {
        note = "同步跟随前请连接已校准的从臂。";
      }
      recordStatus.textContent = error || note;
      const observed = recordingNow ? state.ranges : getObservedLimits();
      const visibleCalibration = calibrationPort.toUpperCase() === mainPort.toUpperCase();
      const leaderRanges = visibleCalibration && (active || leaderStage === "CALIBRATED")
        ? state?.calibration_ranges : recording?.calibration_ranges;
      for (const [joint, columns] of cells) {
        const motor = joint.replace(/\.pos$/, "");
        const followerRaw = state?.follower_calibration_ranges?.[joint] ?? state?.follower_calibration_ranges?.[motor];
        const simJoint = getSimulation()?.joints.find(item => item.name === joint);
        const followerLimit = state?.follower_limits?.[joint] ??
          (simJoint && [simJoint.min, simJoint.max]);
        const liveMin = followerCalibration?.mins_raw?.[motor];
        const liveMax = followerCalibration?.maxes_raw?.[motor];
        columns[0].textContent = rangeLabel(leaderRanges?.[joint] ?? leaderRanges?.[motor], 0);
        const followerRange = followerCalibration?.active || (!followerRaw && followerCalibration?.state === "COMPLETE")
          ? [liveMin, liveMax] : followerRaw;
        columns[1].textContent = rangeLabel(followerRange, 0);
        columns[2].textContent = rangeLabel(followerLimit);
        columns[3].textContent = rangeLabel(observed?.[joint]);
      }
    }

    async function refresh(whilePending = false) {
      if (!opened || (pending && !whilePending)) return;
      const before = revision;
      try {
        const status = await api("/api/teach/status");
        if (before === revision && opened) {
          state = status;
          statusError = "";
        }
      } catch (error) {
        if (error.name !== "AbortError" && before === revision && opened) {
          statusError = error.message;
        }
      }
      if (opened) render();
    }

    async function run(operation, command) {
      if (pending) return false;
      pending = operation;
      ++revision;
      lastError = "";
      render();
      try {
        await command();
        return true;
      } catch (error) {
        if (error.name !== "AbortError") lastError = error.message;
        return false;
      } finally {
        await refresh(true);
        pending = null;
        render();
      }
    }

    function leaderPayload() {
      return { port: leaderPort.value.trim(), follower_port: followerPort.value.trim() };
    }

    function command(operation, payload = {}) {
      return run(operation, () => api(`/api/teach/${operation}`, payload));
    }

    function stopRecording() {
      return run("recording/stop", async () => {
        let result;
        try {
          result = await api("/api/teach/recording/stop", {});
        } catch (error) {
          if (error.status === 410) seenResult = true;
          throw error;
        }
        const { recording, action } = result;
        seenResult = true;
        setRecording(recording, action);
        if (action) {
          name.value = "";
          description.value = "";
          message(`「${action.name}」已保存。可选择回放原始录制，或载入修改动作步骤。`);
        } else {
          message("录制已完成。可回放原始录制或生成动作草稿。");
        }
      });
    }

    leaderPort.addEventListener("input", () => {
      leaderPortEdited = true;
      confirmed.checked = false;
      render();
    });
    followerPort.addEventListener("input", () => {
      confirmed.checked = false;
      follower.setPort(followerPort.value);
    });
    confirmed.addEventListener("change", render);
    name.addEventListener("input", render);
    description.addEventListener("input", render);
    mode.addEventListener("change", () => { confirmed.checked = false; render(); });
    window.addEventListener("so101:follower-state", () => { if (opened) render(); });
    for (const button of leaderCommands) {
      button.addEventListener("click", () => {
        const operation = button.dataset.teachCommand;
        void command(operation, operation === "calibration/start" ? leaderPayload() : {});
      });
    }
    for (const button of followerCommands) {
      button.addEventListener("click", () => void run(button.dataset.followerCommand, async () => {
        const error = await follower.run(button.dataset.followerCommand);
        if (error) throw new Error(error);
      }));
    }
    start.addEventListener("click", () => {
      const sync = mode.value === "sync";
      void run("recording/start", async () => {
        await api("/api/teach/recording/start", {
          ...leaderPayload(), sync, name: name.value.trim(), description: description.value.trim(),
          ...(sync ? {
            mapping_confirmed: confirmed.checked,
            directions: Object.fromEntries([...directions].map(([joint, select]) => [joint, Number(select.value)])),
          } : {}),
        });
        seenResult = false;
      });
    });
    stop.addEventListener("click", () => void stopRecording());
    extract.addEventListener("click", () => void run("draft", async () => {
      const { draft } = await api("/api/simulation/recording/draft", getRecordingPayload());
      applyDraft(draft);
    }));

    return {
      render,
      start() {
        opened = true;
        lastError = "";
        statusError = "";
        confirmed.checked = false;
        render();
        void refresh();
        timer = window.setInterval(() => void refresh(), 600);
      },
      async prepareClose() {
        if (pending) {
          if (pending === "recording/start" || state?.sync_active) {
            await api("/api/disarm", {}, null);
          }
          return !pending;
        }
        if (state?.recording) return stopRecording();
        if (state?.active) return command("calibration/cancel");
        return true;
      },
      close() {
        opened = false;
        ++revision;
        window.clearInterval(timer);
      },
    };
  };
})();
