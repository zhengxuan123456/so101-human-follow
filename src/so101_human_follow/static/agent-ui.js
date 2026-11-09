(() => {
  "use strict";

  const THEME_KEY = "so101-agent-theme";
  const VOICE_KEY = "so101-agent-voice";
  const PREFERENCES_KEY = "so101-agent-preferences";
  const CLIENT_KEY = "so101-human-follow-client-id";
  const AGENT_POLL_MS = 1000;
  const DEFAULT_VOICE = "taiwan_female";
  const MOTION_BUSY_HINT = "上个动作仍在执行或回位，请完成后再发送，或先停止。";
  const NEURAL_VOICES = new Set(["taiwan_female", "puppy_male"]);
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  const elements = {
    openVideoFollowButton: byId("openVideoFollowButton"),
    closeVideoFollowButton: byId("closeVideoFollowButton"),
    videoFollowDialog: byId("videoFollowDialog"),
    videoFollowSteps: byId("videoFollowSteps"),
    videoStopSlot: byId("videoStopSlot"),
    videoDialogStopButton: byId("videoDialogStopButton"),
    openAgentChatButton: byId("openAgentChatButton"),
    closeAgentChatButton: byId("closeAgentChatButton"),
    agentChatDialog: byId("agentChatDialog"),
    pinkThemeButton: byId("pinkThemeButton"),
    whiteThemeButton: byId("whiteThemeButton"),
    openAgentSettingsButton: byId("openAgentSettingsButton"),
    noticeSettingsButton: byId("noticeSettingsButton"),
    settingsDialog: byId("agentSettingsDialog"),
    settingsForm: byId("agentSettingsForm"),
    closeSettingsButton: byId("closeAgentSettingsButton"),
    cancelSettingsButton: byId("cancelAgentSettingsButton"),
    testConnectionButton: byId("testAgentConnectionButton"),
    saveSettingsButton: byId("saveAgentSettingsButton"),
    toggleApiKeyButton: byId("toggleApiKeyButton"),
    agentNameInput: byId("agentNameInput"),
    baseUrlInput: byId("agentBaseUrlInput"),
    apiKeyInput: byId("agentApiKeyInput"),
    modelInput: byId("agentModelInput"),
    voiceSelect: byId("agentVoiceSelect"),
    previewVoiceButton: byId("previewAgentVoiceButton"),
    replaySpeechButton: byId("replaySpeechButton"),
    speechPlaybackMessage: byId("speechPlaybackMessage"),
    settingsMessage: byId("agentSettingsMessage"),
    agentDisplayName: byId("agentDisplayName"),
    agentStatusBadge: byId("agentStatusBadge"),
    agentSetupNotice: byId("agentSetupNotice"),
    chatMessages: byId("agentChatMessages"),
    rawTranscript: byId("rawTranscript"),
    normalizedTranscript: byId("normalizedTranscript"),
    messageForm: byId("agentMessageForm"),
    textInput: byId("agentTextInput"),
    sendButton: byId("sendAgentMessageButton"),
    voiceToggleButton: byId("voiceToggleButton"),
    cancelAgentButton: byId("cancelAgentButton"),
    voiceStateText: byId("voiceStateText"),
  };

  let settingsConfigured = false;
  let agentName = "小助手";
  let agentBusy = false;
  let voiceEnabled = false;
  let recognition = null;
  let recognitionRunning = false;
  let speaking = false;
  let restartTimer = null;
  let statusTimer = null;
  let lastAgentState = "idle";
  let availableVoices = [];
  let remoteAgentStatus = null;
  let remoteMotion = null;
  let localAgentError = "";
  let interactionGeneration = 0;
  let readyTaskId = null;
  let speechGeneration = 0;
  let speechController = null;
  let speechFinish = null;
  let activeAudio = null;
  let activeAudioUrl = null;
  let replaySpeech = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function getClientId() {
    return sessionStorage.getItem(CLIENT_KEY) || "agent-ui";
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-Client-Id", getClientId());
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
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

  function setTheme(theme) {
    const safeTheme = theme === "white" ? "white" : "pink";
    document.documentElement.dataset.theme = safeTheme;
    localStorage.setItem(THEME_KEY, safeTheme);
    for (const button of [elements.pinkThemeButton, elements.whiteThemeButton]) {
      button?.setAttribute("aria-pressed", String(button.dataset.themeChoice === safeTheme));
    }
  }

  function setSettingsMessage(message, kind = "") {
    elements.settingsMessage.textContent = message || "";
    elements.settingsMessage.className = "settings-message";
    if (kind) elements.settingsMessage.classList.add(`is-${kind}`);
  }

  function setAgentBadge(text, kind = "muted") {
    elements.agentStatusBadge.textContent = text;
    elements.agentStatusBadge.className = `badge badge-${kind}`;
  }

  function updateSendButton() {
    elements.sendButton.disabled = agentBusy;
    elements.sendButton.textContent = agentBusy ? "处理中" : remoteMotion?.active ? "动作进行中" : "发送";
    elements.sendButton.title = remoteMotion?.active ? `${MOTION_BUSY_HINT}仍可输入“停止”。` : "";
  }

  function setAgentState(state, detail = "") {
    lastAgentState = state || "idle";
    const states = {
      idle: [settingsConfigured ? "待命" : "未配置", settingsConfigured ? "safe" : "muted", "等待你的指令"],
      listening: ["正在监听", "armed", "正在听你说话…"],
      transcribing: ["正在识别", "warning", "正在整理语音文字…"],
      thinking: ["正在思考", "warning", "正在理解并生成安全动作…"],
      speaking: ["正在回答", "safe", "正在语音回答，结束后会自动继续监听"],
      preparing_speech: ["准备语音", "warning", "正在生成自然语音，准备好后开始播放"],
      awaiting_play: ["点击播放", "warning", "浏览器暂未允许自动播放，请点击下方播放按钮"],
      executing: ["正在执行", "warning", "正在执行已通过安全检查的动作"],
      returning: ["正在回位", "warning", "正在回到本次动作前的位置，可随时停止"],
      waiting_visual: ["等待画面", "warning", "动作已安排，等待摄像头识别到可靠的肩部或手掌…"],
      paused_visual: ["动作已暂停", "warning", "肩部或手掌暂时丢失，正在等待画面恢复…"],
      complete: ["动作流程结束", "safe", "本次动作流程已结束"],
      cancelled: ["动作已取消", "muted", "本次动作已取消，保持当前位置，不会自动回位"],
      error: ["需要处理", "danger", detail || "Agent 发生错误"],
    };
    const [label, kind, description] = states[state] || states.idle;
    setAgentBadge(label, kind);
    elements.voiceStateText.textContent = detail || description;
    elements.cancelAgentButton.disabled = !agentBusy && !remoteMotion?.active &&
      !["thinking", "preparing_speech", "awaiting_play", "speaking", "executing", "returning", "waiting_visual", "paused_visual"].includes(state);
    elements.voiceToggleButton.classList.toggle("is-listening", state === "listening");
    updateSendButton();
  }

  function updateAgentName(value) {
    agentName = String(value || "").trim() || "小助手";
    elements.agentDisplayName.textContent = agentName;
    document.querySelectorAll(".agent-entry-name").forEach((node) => {
      node.textContent = agentName;
    });
    document.querySelectorAll(".assistant-message .chat-speaker").forEach((node) => {
      node.textContent = agentName;
    });
  }

  function appendMessage(role, content, options = {}) {
    const text = String(content || "").trim();
    if (!text) return;

    const article = document.createElement("article");
    article.className = `chat-message ${role === "user" ? "user-message" : "assistant-message"}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "你" : "♡";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    const speaker = document.createElement("strong");
    speaker.className = "chat-speaker";
    speaker.textContent = role === "user" ? "你" : agentName;
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    bubble.append(speaker, paragraph);

    if (options.meta) {
      const meta = document.createElement("small");
      meta.className = "chat-meta";
      meta.textContent = options.meta;
      bubble.append(meta);
    }

    article.append(avatar, bubble);
    elements.chatMessages.append(article);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  }

  function clearStarterMessage() {
    const messages = elements.chatMessages.querySelectorAll(".chat-message");
    if (messages.length === 1 && messages[0].querySelector("p")?.textContent.includes("设置好云端模型")) {
      messages[0].remove();
    }
  }

  function normalizeHistory(payload) {
    if (Array.isArray(payload)) return payload;
    return payload.history || payload.messages || payload.items || [];
  }

  async function loadHistory() {
    try {
      const payload = await api("/api/agent/history");
      const history = normalizeHistory(payload);
      if (!history.length) return;
      elements.chatMessages.replaceChildren();
      for (const item of history) {
        const role = item.role === "user" ? "user" : "assistant";
        appendMessage(role, item.content || item.text || item.reply || "", {
          meta: item.action_summary || "",
        });
      }
    } catch (_error) {
      // A fresh backend may have no history yet; this is not a page-level error.
    }
  }

  function extractSettings(payload) {
    return payload.settings || payload.agent || payload || {};
  }

  function nonSecretPreferences(values) {
    const saved = {};
    // Never serialize the request payload: it can contain an API Key.
    for (const [field, limit] of [["agent_name", 24], ["base_url", 2048], ["model", 120], ["voice", 160]]) {
      const value = values?.[field];
      if (typeof value === "string" && value.trim() && value.length <= limit) {
        saved[field] = value.trim();
      }
    }
    return saved;
  }

  function savedAgentPreferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(PREFERENCES_KEY) || "{}");
      return saved && typeof saved === "object" && !Array.isArray(saved)
        ? nonSecretPreferences(saved) : {};
    } catch (_error) {
      return {};
    }
  }

  function rememberAgentPreferences(values) {
    const saved = nonSecretPreferences(values);
    try {
      localStorage.setItem(PREFERENCES_KEY, JSON.stringify(saved));
      if (saved.voice) localStorage.setItem(VOICE_KEY, saved.voice);
    } catch (_error) {
      // Backend settings are already saved.  A blocked browser backup must not
      // turn that successful save into an error or retain the Key in the form.
    }
  }

  function applySettings(payload) {
    const settings = extractSettings(payload);
    const saved = savedAgentPreferences();
    const configured = Boolean(
      payload.configured ?? settings.configured ?? settings.api_key_set ?? settings.has_api_key ?? settings.api_key_configured,
    );
    settingsConfigured = configured;
    updateAgentName(settings.agent_name || settings.name || saved.agent_name || agentName);
    elements.agentNameInput.value = settings.agent_name || settings.name || saved.agent_name || agentName;
    elements.baseUrlInput.value = settings.base_url || settings.api_base || saved.base_url || "";
    elements.modelInput.value = settings.model || saved.model || "";
    elements.apiKeyInput.value = "";
    let legacyVoice = "";
    try { legacyVoice = localStorage.getItem(VOICE_KEY) || ""; } catch (_error) { /* optional backup */ }
    const selectedVoice = settings.voice || saved.voice || legacyVoice || DEFAULT_VOICE;
    if (selectedVoice) elements.voiceSelect.value = selectedVoice;
    elements.agentSetupNotice.hidden = configured;
    if (!agentBusy && !speaking && !voiceEnabled) setAgentState("idle");
  }

  async function loadSettings() {
    try {
      const payload = await api("/api/agent/settings");
      applySettings(payload);
    } catch (error) {
      applySettings({ configured: false });
      setAgentState("error", `无法读取 Agent 设置：${error.message}`);
    }
  }

  function settingsPayload() {
    const payload = {
      agent_name: elements.agentNameInput.value.trim(),
      base_url: elements.baseUrlInput.value.trim(),
      model: elements.modelInput.value.trim(),
      voice: elements.voiceSelect.value,
    };
    const apiKey = elements.apiKeyInput.value.trim();
    if (apiKey) payload.api_key = apiKey;
    return payload;
  }

  function validateSettings(payload) {
    if (!payload.agent_name) return "请填写 Agent 名字。";
    if (!payload.base_url) return "请填写 API 地址。";
    try {
      const url = new URL(payload.base_url);
      if (!/^https?:$/.test(url.protocol)) return "API 地址只支持 http 或 https。";
    } catch (_error) {
      return "API 地址格式不正确。";
    }
    if (!payload.model) return "请填写模型名称。";
    if (!settingsConfigured && !payload.api_key) return "首次设置需要填写 API Key。";
    return "";
  }

  async function saveSettings(event) {
    event.preventDefault();
    const payload = settingsPayload();
    const validation = validateSettings(payload);
    if (validation) {
      setSettingsMessage(validation, "error");
      return;
    }
    setSettingsBusy(true);
    setSettingsMessage("正在安全保存设置…");
    try {
      const result = await api("/api/agent/settings", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const savedSettings = extractSettings(result);
      rememberAgentPreferences({
        agent_name: savedSettings.agent_name || payload.agent_name,
        base_url: savedSettings.base_url || payload.base_url,
        model: savedSettings.model || payload.model,
        voice: savedSettings.voice || payload.voice,
      });
      applySettings(result.settings ? result : { ...payload, configured: true });
      settingsConfigured = true;
      elements.agentSetupNotice.hidden = true;
      elements.apiKeyInput.value = "";
      setSettingsMessage("设置已保存：名字、API 地址、模型和音色下次自动填入；API Key 只保存在本次后端内存中。", "success");
      setAgentState("idle");
      window.setTimeout(() => elements.settingsDialog.close(), 650);
    } catch (error) {
      setSettingsMessage(error.message, "error");
    } finally {
      setSettingsBusy(false);
    }
  }

  async function testConnection() {
    const payload = settingsPayload();
    const validation = validateSettings(payload);
    if (validation) {
      setSettingsMessage(validation, "error");
      return;
    }
    setSettingsBusy(true);
    setSettingsMessage("正在连接云端模型并检查 Function Calling…");
    try {
      const result = await api("/api/agent/test-connection", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSettingsMessage(result.message || "连接成功，模型支持当前 Agent 配置。", "success");
    } catch (error) {
      setSettingsMessage(error.message, "error");
    } finally {
      setSettingsBusy(false);
    }
  }

  function setSettingsBusy(value) {
    elements.testConnectionButton.disabled = value;
    elements.saveSettingsButton.disabled = value;
  }

  function openSettings() {
    setSettingsMessage("");
    if (!elements.settingsDialog.open) elements.settingsDialog.showModal();
  }

  function isLocalStopCommand(text) {
    const cleaned = String(text || "")
      .replace(/[，。！？!?,.\s]/g, "")
      .replace(new RegExp(`^${escapeRegExp(agentName)}`), "");
    return /^(停止|停下|别动|不要动|马上停止|紧急停止|急停)$/.test(cleaned);
  }

  function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  async function executeLocalStop(sourceText) {
    invalidateInteraction();
    const generation = interactionGeneration;
    agentBusy = true;
    localAgentError = "";
    clearStarterMessage();
    appendMessage("user", sourceText);
    setAgentState("executing", "识别到本地停止词，正在立即停止动作…");
    try {
      const result = await api("/api/disarm", { method: "POST" });
      if (generation !== interactionGeneration) return;
      const reply = result.message || "已经停止动作。";
      appendMessage("assistant", reply, { meta: "本地停止指令 · 未调用云端模型" });
      await speakReply(reply);
    } catch (error) {
      appendMessage("assistant", `停止请求失败：${error.message}`);
      setAgentState("error", error.message);
    } finally {
      if (generation === interactionGeneration) {
        agentBusy = false;
        if (!speaking) resumeListeningOrIdle();
      }
    }
  }

  function extractReply(payload) {
    return payload.reply || payload.assistant_reply || payload.response || payload.message || "指令已处理。";
  }

  function extractNormalized(payload, fallback) {
    return payload.normalized_text || payload.normalized || payload.command || fallback;
  }

  function actionMeta(payload) {
    if (payload.action_summary) return payload.action_summary;
    const actions = payload.actions || payload.plan?.actions || [];
    if (!Array.isArray(actions) || !actions.length) return "";
    const labels = {
      enter_pet_pose: "低头停留2秒并回位",
      leave_pet_pose: "小幅抬头",
      return_to_action_start: "回到刚才的位置",
      open_gripper: "张开夹爪",
      close_gripper: "合上夹爪",
      return_to_neutral: "返回起始姿势",
      wait: "等待",
    };
    const responseLabels = {
      neutral: "自然回应并回位",
      happy: "开心回应并回位",
      curious: "好奇回应并回位",
      cute: "撒娇回应并回位",
      encouraging: "鼓励回应并回位",
    };
    const gestureLabels = {
      nod: "点头",
      shake_head: "摇头",
      wave: "腕部挥动",
      wiggle: "轻轻摆动",
    };
    return `已安排动作：${actions.map((item) => {
      const name = item.name || item.action;
      if (name === "perform_greeting") {
        const count = Number(item.arguments?.count ?? 2);
        const countLabel = { 1: "一", 2: "两", 3: "三" }[count] || "两";
        return `夹爪张合${countLabel}次`;
      }
      if (name === "perform_response_gesture") {
        return responseLabels[item.arguments?.category] || "通用回应并回位";
      }
      if (name === "perform_gesture") {
        const gesture = gestureLabels[item.arguments?.gesture] || "姿态回应";
        const count = Number(item.arguments?.count);
        return `${gesture}${Number.isInteger(count) && count > 0 ? `${count}次` : ""}并回位`;
      }
      return labels[name] || name;
    }).filter(Boolean).join(" → ")}`;
  }

  function isMotionBusyError(message) {
    const text = String(message || "");
    return text.includes("Another Agent motion is already active") ||
      text.includes("上一个机械臂动作仍在执行或回位");
  }

  function friendlyAgentError(message) {
    const text = String(message || "");
    if (isMotionBusyError(text)) {
      return MOTION_BUSY_HINT;
    }
    if (text.includes("Connect the robot before starting an Agent motion")) {
      return "机械臂尚未连接，请在首页连接从臂后再发送动作指令。";
    }
    if (text.includes("UNKNOWN") || text.includes("visual proximity is stale") ||
        text.includes("visual recognition timed out") || text.includes("visual safety wait timed out")) {
      return "未获得可靠的肩部或手掌画面，动作已停止；让肩部或手掌进入电脑摄像头画面后再试。";
    }
    if (text.includes("TOO_CLOSE") || text.includes("too close")) {
      return "画面中的人或手过近，动作已停止；调整位置后再试。";
    }
    if (text.includes("browser heartbeat") || text.includes("control lease")) {
      return "页面控制连接已中断，动作已停止；请只保留一个控制台页面并刷新。";
    }
    if (text.includes("operator cancelled") || text.includes("operator pressed stop-following")) {
      return "你已取消本次动作，保持当前位置，不会自动回位。";
    }
    if (text.includes("return position") || text.includes("restore position") || text.includes("return-to-start")) {
      return "回位未完成，动作已停止并保持当前位置；请检查实际姿势后再操作。";
    }
    return text;
  }

  function renderMotionStatus() {
    if (!remoteMotion) return false;
    const state = String(remoteMotion.state || "").toUpperCase();
    if (remoteMotion.active) {
      if (state === "WAITING_VISUAL") {
        setAgentState("waiting_visual", "动作已安排，等待可靠的肩部或手掌画面；识别恢复后自动开始，可随时取消。");
      } else if (state === "PAUSED_VISUAL") {
        setAgentState("paused_visual", "视觉识别暂时丢失，已暂停发送动作；画面恢复后继续，超时会取消。");
      } else if (state === "RETURNING") {
        const total = Number(remoteMotion.step_count) || 0;
        const index = (Number(remoteMotion.step_index) || 0) + 1;
        setAgentState("returning", total ? `正在回位（步骤 ${index}/${total}）。${MOTION_BUSY_HINT}` : MOTION_BUSY_HINT);
      } else {
        const total = Number(remoteMotion.step_count) || 0;
        const index = (Number(remoteMotion.step_index) || 0) + 1;
        setAgentState("executing", total ? `正在执行动作步骤 ${index}/${total}。${MOTION_BUSY_HINT}` : MOTION_BUSY_HINT);
      }
      return true;
    }
    // A new cloud request takes precedence over a previous motion's terminal state.
    const task = remoteAgentStatus?.latest_task;
    const taskState = String(task?.state || "").toUpperCase();
    if (["QUEUED", "THINKING", "READY", "DISPATCHING", "FAILED", "WAITING_CONFIRMATION"].includes(taskState)) {
      return false;
    }
    if (state === "COMPLETE" && !localAgentError) {
      setAgentState("complete", remoteMotion.return_verified
        ? "动作已完成，并已确认回到本次动作前的位置。"
        : "动作已完成，停在本次目标姿势；其他指令可继续使用。");
      return true;
    }
    if (["CANCELLED", "FAULT"].includes(state) && !localAgentError) {
      const reason = friendlyAgentError(remoteMotion.error || remoteMotion.message);
      setAgentState(state === "FAULT" ? "error" : "cancelled", reason);
      return true;
    }
    return false;
  }

  function appendConfirmationControls(taskId) {
    if (!taskId) return;
    const row = document.createElement("div");
    row.className = "agent-confirmation-row";
    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.className = "button button-agent";
    confirmButton.textContent = "确认执行";
    const rejectButton = document.createElement("button");
    rejectButton.type = "button";
    rejectButton.className = "button button-danger-outline";
    rejectButton.textContent = "取消";
    row.append(confirmButton, rejectButton);
    elements.chatMessages.append(row);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    const finish = () => {
      confirmButton.disabled = true;
      rejectButton.disabled = true;
      row.remove();
    };
    confirmButton.addEventListener("click", async () => {
      confirmButton.disabled = true;
      rejectButton.disabled = true;
      setAgentState("executing", "正在执行你刚刚确认的动作…");
      try {
        const result = await api("/api/agent/confirm", {
          method: "POST",
          body: JSON.stringify({ task_id: taskId }),
        });
        appendMessage("assistant", extractReply(result), {
          meta: actionMeta(result) || "已确认动作计划，等待执行状态更新",
        });
        await pollAgentStatus();
      } catch (error) {
        appendMessage("assistant", `确认执行失败：${error.message}`);
        setAgentState("error", error.message);
      } finally {
        finish();
        resumeListeningOrIdle();
      }
    });
    rejectButton.addEventListener("click", async () => {
      try {
        await api("/api/agent/cancel", {
          method: "POST",
          body: JSON.stringify({ task_id: taskId }),
        });
        appendMessage("assistant", "好的，这个动作已取消。");
      } catch (error) {
        appendMessage("assistant", `取消失败：${error.message}`);
      } finally {
        finish();
        resumeListeningOrIdle();
      }
    });
  }

  async function sendMessage(text, source = "text") {
    const cleanText = String(text || "").trim();
    if (!cleanText) return;
    if (isLocalStopCommand(cleanText)) {
      await executeLocalStop(cleanText);
      return;
    }
    if (agentBusy) return;
    if (!settingsConfigured) {
      setAgentState("error", "请先完成 Agent 设置并测试云端模型。 ");
      openSettings();
      return;
    }

    const generation = ++interactionGeneration;
    let deferredTaskId = null;
    let motionDispatched = false;
    agentBusy = true;
    localAgentError = "";
    // Keep continuous voice recognition alive while the cloud model is thinking
    // so a local STOP word can still pre-empt the in-flight request.  Non-stop
    // utterances are ignored by the agentBusy gate below; speech is paused later
    // while the assistant is speaking to avoid recognizing its own reply.
    if (!voiceEnabled) pauseRecognition();
    updateSendButton();

    try {
      // Speech can finish before the robot has returned. Check fresh physical
      // motion state before clearing text or starting another cloud request.
      await pollAgentStatus({ strict: true });
      if (generation !== interactionGeneration) return;
      if (remoteMotion?.active) {
        renderMotionStatus();
        elements.voiceStateText.textContent = MOTION_BUSY_HINT;
        return;
      }
      clearStarterMessage();
      appendMessage("user", cleanText);
      elements.rawTranscript.textContent = cleanText;
      elements.normalizedTranscript.textContent = "正在规范化…";
      elements.textInput.value = "";
      elements.cancelAgentButton.disabled = false;
      setAgentState("thinking");
      const payload = await api("/api/agent/message", {
        method: "POST",
        body: JSON.stringify({ text: cleanText, source, defer_motion: true }),
      });
      if (generation !== interactionGeneration) return;
      elements.normalizedTranscript.textContent = extractNormalized(payload, cleanText);
      const state = String(payload.state || payload.status || "").toLowerCase();
      if (state === "cancelled") {
        appendMessage("assistant", "这条指令已取消，没有执行任何动作。", {
          meta: "本地停止已生效",
        });
        return;
      }
      const reply = extractReply(payload);
      appendMessage("assistant", reply, { meta: actionMeta(payload) });
      if (payload.requires_confirmation || state === "waiting_confirmation") {
        appendConfirmationControls(payload.task_id);
      }
      const scheduledMotion = payload.result?.agent_motion;
      if (scheduledMotion) remoteMotion = { ...scheduledMotion, active: true };
      readyTaskId = state === "ready" ? payload.task_id : null;
      void pollAgentStatus();
      if (state.includes("execut")) setAgentState("executing");
      deferredTaskId = readyTaskId;
      await speakReply(reply, {
        onPlaybackStart: deferredTaskId ? async (duration, signal) => {
          if (generation !== interactionGeneration || readyTaskId !== deferredTaskId) {
            throw new DOMException("Interaction cancelled", "AbortError");
          }
          const started = await api("/api/agent/start", {
            method: "POST",
            signal,
            body: JSON.stringify({ task_id: deferredTaskId, speech_duration_s: duration }),
          });
          if (generation !== interactionGeneration) return;
          motionDispatched = true;
          readyTaskId = null;
          const motion = started.result?.agent_motion || started.agent_motion;
          if (motion) remoteMotion = { ...motion, active: true };
          void pollAgentStatus();
        } : null,
      });
    } catch (error) {
      if (generation !== interactionGeneration || error.name === "AbortError") return;
      if (deferredTaskId) {
        readyTaskId = null;
        await api("/api/agent/cancel", {
          method: "POST",
          body: JSON.stringify({ task_id: deferredTaskId, discard_only: !motionDispatched }),
        }).catch(() => {});
      }
      if (isMotionBusyError(error.message)) {
        // Another page may have started a motion after our status preflight.
        // Discard only this prepared request and leave that motion untouched.
        if (source === "text" && !elements.textInput.value.trim()) elements.textInput.value = cleanText;
        localAgentError = "";
        await pollAgentStatus();
        renderMotionStatus();
        elements.voiceStateText.textContent = MOTION_BUSY_HINT;
        return;
      }
      if (elements.normalizedTranscript.textContent === "正在规范化…") {
        elements.normalizedTranscript.textContent = cleanText;
      }
      const reason = friendlyAgentError(error.message);
      localAgentError = reason;
      appendMessage("assistant", `这次没有执行：${reason}`);
      setAgentState("error", reason);
    } finally {
      if (generation === interactionGeneration) {
        agentBusy = false;
        updateSendButton();
        elements.cancelAgentButton.disabled = !remoteMotion?.active && !readyTaskId;
        if (!speaking) resumeListeningOrIdle();
      }
    }
  }

  async function cancelInteraction() {
    const pendingTaskId = readyTaskId;
    invalidateInteraction();
    try {
      await api("/api/agent/cancel", {
        method: "POST",
        body: JSON.stringify(pendingTaskId ? { task_id: pendingTaskId } : {}),
      });
      localAgentError = "";
      remoteMotion = null;
      appendMessage("assistant", "当前交互已取消。机械臂保持安全状态。");
      await pollAgentStatus();
    } catch (error) {
      appendMessage("assistant", `取消失败：${error.message}`);
    } finally {
      agentBusy = false;
      elements.cancelAgentButton.disabled = true;
      resumeListeningOrIdle();
    }
  }

  function populateVoices() {
    availableVoices = window.speechSynthesis?.getVoices() || [];
    const current = localStorage.getItem(VOICE_KEY) || elements.voiceSelect.value || DEFAULT_VOICE;
    elements.voiceSelect.replaceChildren(
      new Option("台妹女声 · 晓雨（自然语速）", "taiwan_female"),
      new Option("小奶狗男声 · 年轻清朗云希", "puppy_male"),
      new Option("本机中文语音（离线）", "system_auto"),
    );
    const sorted = [...availableVoices].sort((a, b) => {
      const aChinese = /^zh/i.test(a.lang) ? 0 : 1;
      const bChinese = /^zh/i.test(b.lang) ? 0 : 1;
      return aChinese - bChinese || a.name.localeCompare(b.name, "zh-CN");
    });
    for (const voice of sorted) {
      elements.voiceSelect.add(new Option(`本机：${voice.name} (${voice.lang})`, voice.name));
    }
    if ([...elements.voiceSelect.options].some((option) => option.value === current)) {
      elements.voiceSelect.value = current;
    }
  }

  function selectedVoice(preset) {
    const selected = preset || elements.voiceSelect.value || localStorage.getItem(VOICE_KEY) || DEFAULT_VOICE;
    if (selected === "taiwan_female") {
      const taiwanVoice = availableVoices.find((voice) => /^zh(-|_)?TW/i.test(voice.lang));
      if (taiwanVoice) return taiwanVoice;
    }
    if (selected === "puppy_male") {
      const maleVoice = availableVoices.find((voice) => /^zh/i.test(voice.lang) &&
        /yunxi|yunyang|yunjian|yunfeng|kangkang|云希|云扬|云健|康康/i.test(voice.name));
      if (maleVoice) return maleVoice;
    }
    return availableVoices.find((voice) => voice.name === selected) ||
      availableVoices.find((voice) => /^zh(-|_)?CN/i.test(voice.lang)) ||
      availableVoices.find((voice) => /^zh/i.test(voice.lang)) || null;
  }

  function releaseAudio() {
    if (activeAudio) {
      activeAudio.onended = null;
      activeAudio.onerror = null;
      activeAudio.onplaying = null;
      activeAudio.pause();
      activeAudio.removeAttribute("src");
      activeAudio.load();
      activeAudio = null;
    }
    if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = null;
  }

  function stopSpeech() {
    speechGeneration += 1;
    speechController?.abort();
    speechController = null;
    releaseAudio();
    window.speechSynthesis?.cancel();
    const finish = speechFinish;
    speechFinish = null;
    finish?.();
    speaking = false;
    replaySpeech = null;
    elements.replaySpeechButton.hidden = true;
  }

  function invalidateInteraction() {
    interactionGeneration += 1;
    readyTaskId = null;
    stopSpeech();
    agentBusy = false;
    updateSendButton();
  }

  async function speakReply(text, { onPlaybackStart = null } = {}) {
    const cleanText = String(text || "").trim();
    if (!cleanText) return;
    stopSpeech();
    const generation = speechGeneration;
    const controller = new AbortController();
    speechController = controller;
    const voice = elements.voiceSelect.value || localStorage.getItem(VOICE_KEY) || DEFAULT_VOICE;
    const isCurrent = () => generation === speechGeneration && !controller.signal.aborted;
    let startPromise = null;
    let playbackStarted = false;
    let resolveCompletion;
    let rejectCompletion;
    const completion = new Promise((resolve, reject) => {
      resolveCompletion = resolve;
      rejectCompletion = reject;
    });
    // A STOP resolves this pending await as well as aborting synthesis/playback.
    speechFinish = resolveCompletion;
    const finish = (error) => {
      if (!isCurrent()) return;
      speechFinish = null;
      releaseAudio();
      controller.abort();
      speechController = null;
      speaking = false;
      replaySpeech = null;
      elements.replaySpeechButton.hidden = true;
      if (error) rejectCompletion(error);
      else resolveCompletion();
      if (!agentBusy) resumeListeningOrIdle();
    };
    const durationEstimate = Math.max(0.5, Math.min(30, cleanText.length / 4.8));
    const onActualStart = (duration) => {
      if (!isCurrent() || playbackStarted) return;
      playbackStarted = true;
      elements.replaySpeechButton.hidden = true;
      setAgentState("speaking");
      const seconds = Number.isFinite(duration) && duration > 0 ? duration : durationEstimate;
      startPromise = Promise.resolve()
        .then(() => {
          if (!isCurrent()) return;
          return onPlaybackStart?.(seconds, controller.signal);
        })
        .catch((error) => {
          // Once playback has started, do not retry and launch the motion twice.
          if (isCurrent()) finish(error);
        });
    };
    const onEnd = () => {
      void Promise.resolve(startPromise).then(() => {
        if (isCurrent()) finish();
      });
    };
    const showPlayButton = (play) => {
      if (!isCurrent()) return;
      replaySpeech = play;
      elements.replaySpeechButton.textContent = onPlaybackStart ? "点击播放并开始动作" : "点击播放语音";
      elements.replaySpeechButton.hidden = false;
      if (elements.settingsDialog.open) {
        elements.previewVoiceButton.disabled = false;
        elements.previewVoiceButton.textContent = "点击播放试听";
      }
      elements.speechPlaybackMessage.textContent = onPlaybackStart
        ? "浏览器阻止了自动播放。点击播放后才会开始本次机械臂动作。"
        : "浏览器阻止了自动播放，请点击播放语音。";
      setAgentState("awaiting_play");
    };

    pauseRecognition();
    speaking = true;
    elements.speechPlaybackMessage.textContent = "";
    setAgentState("preparing_speech");

    const playLocalVoice = (fallbackReason = "") => {
      if (!isCurrent()) return;
      if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
        finish(new Error("当前浏览器无法播放语音；本次动作没有启动，请更换浏览器重试。"));
        return;
      }
      const localVoice = selectedVoice(voice);
      if (fallbackReason) {
        elements.speechPlaybackMessage.textContent = `自然音色暂时不可用，已切换本机语音${localVoice ? `（${localVoice.name}）` : ""}。`;
      }
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.lang = localVoice?.lang || "zh-CN";
      if (localVoice) utterance.voice = localVoice;
      utterance.rate = voice === "taiwan_female" ? 1.0 : 1.08;
      utterance.pitch = 1;
      utterance.onstart = () => onActualStart(durationEstimate);
      utterance.onend = onEnd;
      utterance.onerror = (event) => {
        if (!isCurrent()) return;
        if (event.error === "not-allowed" && !playbackStarted) {
          showPlayButton(() => playLocalVoice(fallbackReason));
        } else {
          finish(new Error("本机语音播放失败，本次交互已停止。"));
        }
      };
      window.speechSynthesis.speak(utterance);
    };

    const prepareNeuralVoice = async () => {
      const response = await fetch("/api/agent/speech", {
        method: "POST",
        credentials: "same-origin",
        cache: "no-store",
        signal: controller.signal,
        headers: { "Content-Type": "application/json", "X-Client-Id": getClientId() },
        body: JSON.stringify({ text: cleanText, voice }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `语音生成失败：${response.status}`);
      }
      const blob = await response.blob();
      if (!isCurrent()) return;
      if (!blob.size) throw new Error("语音服务返回了空音频。 ");
      activeAudioUrl = URL.createObjectURL(blob);
      const audio = new Audio(activeAudioUrl);
      activeAudio = audio;
      audio.preload = "auto";
      await new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("音频加载超时。")), 5000);
        const settle = (callback) => {
          window.clearTimeout(timeout);
          audio.onloadedmetadata = null;
          audio.onerror = null;
          callback();
        };
        audio.onloadedmetadata = () => settle(resolve);
        audio.onerror = () => settle(() => reject(new Error("音频无法加载。")));
        if (audio.readyState >= 1) settle(resolve);
      });
      if (!isCurrent()) return;
      audio.onplaying = () => onActualStart(audio.duration);
      audio.onended = onEnd;
      audio.onerror = () => finish(new Error("语音播放中断，本次交互已停止。"));
      const play = async () => {
        if (!isCurrent()) return;
        try {
          await audio.play();
        } catch (error) {
          if (!isCurrent()) return;
          if (error.name === "NotAllowedError") showPlayButton(play);
          else finish(new Error("语音无法播放，本次交互已停止。"));
        }
      };
      await play();
    };

    if (NEURAL_VOICES.has(voice)) {
      try {
        await prepareNeuralVoice();
      } catch (error) {
        if (isCurrent() && error.name !== "AbortError") {
          releaseAudio();
          console.warn("Neural voice unavailable; using system voice", error.message);
          playLocalVoice(error.message);
        }
      }
    } else {
      playLocalVoice();
    }
    return completion;
  }

  async function previewVoice() {
    if (replaySpeech && elements.settingsDialog.open) {
      const play = replaySpeech;
      replaySpeech = null;
      elements.previewVoiceButton.disabled = true;
      void play();
      return;
    }
    if (agentBusy || readyTaskId || remoteMotion?.active) {
      setSettingsMessage("请先结束当前交互，再试听音色。", "error");
      return;
    }
    elements.previewVoiceButton.disabled = true;
    setSettingsMessage("正在准备试听，只播放声音，不执行动作…");
    try {
      await speakReply(`你好呀，我是${elements.agentNameInput.value.trim() || agentName}，很高兴见到你。`);
      setSettingsMessage("试听结束，可以选择喜欢的音色后保存设置。", "success");
    } catch (error) {
      setSettingsMessage(error.message, "error");
    } finally {
      elements.previewVoiceButton.disabled = false;
      elements.previewVoiceButton.textContent = "试听所选音色";
    }
  }

  function createRecognition() {
    if (!SpeechRecognition) return null;
    const instance = new SpeechRecognition();
    instance.lang = "zh-CN";
    instance.continuous = true;
    instance.interimResults = true;
    instance.maxAlternatives = 1;

    instance.onstart = () => {
      recognitionRunning = true;
      if (voiceEnabled && !speaking && !agentBusy && !renderMotionStatus()) setAgentState("listening");
    };
    instance.onresult = (event) => {
      let interimText = "";
      const finalParts = [];
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index][0]?.transcript || "";
        if (event.results[index].isFinal) finalParts.push(transcript);
        else interimText += transcript;
      }
      if (interimText) elements.rawTranscript.textContent = interimText;
      const finalText = finalParts.join("").trim();
      if (finalText) {
        elements.rawTranscript.textContent = finalText;
        setAgentState("transcribing");
        void sendMessage(finalText, "voice");
      }
    };
    instance.onerror = (event) => {
      if (["aborted", "no-speech"].includes(event.error)) return;
      if (["not-allowed", "service-not-allowed"].includes(event.error)) {
        voiceEnabled = false;
        updateVoiceButton();
        setAgentState("error", "浏览器没有麦克风权限，请允许访问后重试。");
        return;
      }
      setAgentState("error", `语音识别失败：${event.error}`);
    };
    instance.onend = () => {
      recognitionRunning = false;
      scheduleRecognitionRestart();
    };
    return instance;
  }

  function startRecognition() {
    // Keep the microphone session alive while the cloud request is pending.  The
    // message gate ignores ordinary extra utterances, but STOP words must remain
    // available until the operator explicitly turns voice mode off.
    if (!voiceEnabled || speaking || recognitionRunning || !recognition) return;
    window.clearTimeout(restartTimer);
    try {
      recognition.start();
    } catch (error) {
      if (error.name !== "InvalidStateError") {
        setAgentState("error", `无法启动语音识别：${error.message}`);
      }
    }
  }

  function pauseRecognition() {
    window.clearTimeout(restartTimer);
    if (!recognition || !recognitionRunning) return;
    try {
      recognition.abort();
    } catch (_error) {
      // Recognition may already be ending.
    }
  }

  function scheduleRecognitionRestart() {
    window.clearTimeout(restartTimer);
    if (!voiceEnabled || speaking) return;
    restartTimer = window.setTimeout(startRecognition, 280);
  }

  function resumeListeningOrIdle() {
    if (renderMotionStatus()) {
      if (voiceEnabled) scheduleRecognitionRestart();
      return;
    }
    if (localAgentError || remoteAgentStatus?.state === "error") {
      setAgentState("error", localAgentError || friendlyAgentError(remoteAgentStatus.message));
      if (voiceEnabled) scheduleRecognitionRestart();
      return;
    }
    if (voiceEnabled) {
      setAgentState("listening");
      scheduleRecognitionRestart();
    } else {
      setAgentState("idle");
    }
  }

  function updateVoiceButton() {
    elements.voiceToggleButton.setAttribute("aria-pressed", String(voiceEnabled));
    const strong = elements.voiceToggleButton.querySelector("strong");
    const small = elements.voiceToggleButton.querySelector("small");
    if (voiceEnabled) {
      strong.textContent = "停止语音 Agent";
      small.textContent = "当前会话持续监听中";
    } else {
      strong.textContent = "开始语音 Agent";
      small.textContent = "开启后持续监听，直到手动关闭";
      elements.voiceToggleButton.classList.remove("is-listening");
    }
  }

  function toggleVoice() {
    if (!SpeechRecognition) {
      setAgentState("error", "当前浏览器不支持持续语音识别，请使用支持 SpeechRecognition 的浏览器。 ");
      return;
    }
    if (!settingsConfigured) {
      openSettings();
      setAgentState("error", "请先配置云端模型。 ");
      return;
    }
    voiceEnabled = !voiceEnabled;
    updateVoiceButton();
    if (voiceEnabled) {
      startRecognition();
      if (!renderMotionStatus()) setAgentState("listening");
    } else {
      pauseRecognition();
      setAgentState("idle");
    }
  }

  function renderRemoteStatus(payload) {
    const status = payload.status || payload.agent || payload || {};
    remoteAgentStatus = status;
    remoteMotion = status.motion || null;
    const remoteName = status.agent_name || status.name;
    if (remoteName) updateAgentName(remoteName);
    const configured = status.configured ?? status.api_key_set;
    if (configured !== undefined) {
      settingsConfigured = Boolean(configured);
      elements.agentSetupNotice.hidden = settingsConfigured;
    }
    const remoteState = String(status.state || "idle").toLowerCase();
    elements.cancelAgentButton.disabled = !agentBusy && !speaking && !remoteMotion?.active;
    updateSendButton();
    if (agentBusy || speaking) return;
    if (renderMotionStatus()) return;
    if (localAgentError) {
      setAgentState("error", localAgentError);
      return;
    }
    if (voiceEnabled) {
      if (["thinking", "speaking", "executing", "error"].includes(remoteState)) {
        setAgentState(remoteState, friendlyAgentError(status.message));
      } else {
        setAgentState("listening");
      }
      return;
    }
    if (["idle", "thinking", "speaking", "executing", "error"].includes(remoteState)) {
      setAgentState(remoteState, friendlyAgentError(status.message));
    }
  }

  async function pollAgentStatus({ strict = false } = {}) {
    try {
      const payload = await api("/api/agent/status");
      renderRemoteStatus(payload);
    } catch (error) {
      // Settings load already provides a visible error. Keep robot controls usable
      // even while the optional Agent backend is unavailable.
      if (strict) throw error;
    }
  }

  function bindEvents() {
    elements.openVideoFollowButton.addEventListener("click", () => openModeDialog(elements.videoFollowDialog));
    elements.closeVideoFollowButton.addEventListener("click", () => elements.videoFollowDialog.close());
    elements.videoDialogStopButton.addEventListener("click", () => byId("stopButton").click());
    elements.openAgentChatButton.addEventListener("click", () => openModeDialog(elements.agentChatDialog));
    elements.closeAgentChatButton.addEventListener("click", () => elements.agentChatDialog.close());
    for (const dialog of [elements.videoFollowDialog, elements.agentChatDialog]) {
      dialog.addEventListener("click", (event) => {
        if (event.target === dialog) dialog.close();
      });
    }
    elements.pinkThemeButton.addEventListener("click", () => setTheme("pink"));
    elements.whiteThemeButton.addEventListener("click", () => setTheme("white"));
    elements.openAgentSettingsButton.addEventListener("click", openSettings);
    elements.noticeSettingsButton.addEventListener("click", openSettings);
    elements.closeSettingsButton.addEventListener("click", () => elements.settingsDialog.close());
    elements.cancelSettingsButton.addEventListener("click", () => elements.settingsDialog.close());
    elements.settingsDialog.addEventListener("click", (event) => {
      if (event.target === elements.settingsDialog) elements.settingsDialog.close();
    });
    elements.toggleApiKeyButton.addEventListener("click", () => {
      const visible = elements.apiKeyInput.type === "text";
      elements.apiKeyInput.type = visible ? "password" : "text";
      elements.toggleApiKeyButton.textContent = visible ? "显示" : "隐藏";
    });
    elements.settingsForm.addEventListener("submit", saveSettings);
    elements.testConnectionButton.addEventListener("click", testConnection);
    elements.previewVoiceButton.addEventListener("click", () => void previewVoice());
    elements.replaySpeechButton.addEventListener("click", () => {
      const play = replaySpeech;
      if (play) void play();
    });
    elements.messageForm.addEventListener("submit", (event) => {
      event.preventDefault();
      void sendMessage(elements.textInput.value, "text");
    });
    elements.voiceToggleButton.addEventListener("click", toggleVoice);
    elements.cancelAgentButton.addEventListener("click", () => void cancelInteraction());
    elements.voiceSelect.addEventListener("change", () => {
      localStorage.setItem(VOICE_KEY, elements.voiceSelect.value);
      stopSpeech();
    });
    byId("stopButton")?.addEventListener("click", invalidateInteraction);
    window.addEventListener("pagehide", () => {
      voiceEnabled = false;
      pauseRecognition();
      invalidateInteraction();
    });
  }

  function openModeDialog(dialog) {
    for (const other of [elements.videoFollowDialog, elements.agentChatDialog]) {
      if (other !== dialog && other.open) other.close();
    }
    if (!dialog.open) dialog.showModal();
  }

  function mountModeControls() {
    // Move, do not clone: the original app.js listeners and live state stay on
    // the same DOM nodes while the home page remains intentionally compact.
    document.querySelectorAll(".visual-follow-step").forEach((step) => {
      elements.videoFollowSteps.append(step);
    });
    const stopButton = byId("stopButton");
    if (stopButton) {
      const syncStopAvailability = () => {
        elements.videoDialogStopButton.disabled = stopButton.disabled;
      };
      syncStopAvailability();
      new MutationObserver(syncStopAvailability).observe(stopButton, {
        attributes: true,
        attributeFilter: ["disabled"],
      });
    }
  }

  function init() {
    setTheme(localStorage.getItem(THEME_KEY) || "pink");
    mountModeControls();
    bindEvents();
    populateVoices();
    if (window.speechSynthesis) {
      speechSynthesis.addEventListener("voiceschanged", populateVoices);
    }
    recognition = createRecognition();
    if (!recognition) {
      elements.voiceToggleButton.title = "当前浏览器不支持 SpeechRecognition，仍可使用文字对话";
    }
    void loadSettings();
    void loadHistory();
    void pollAgentStatus();
    statusTimer = window.setInterval(pollAgentStatus, AGENT_POLL_MS);
  }

  init();
})();
