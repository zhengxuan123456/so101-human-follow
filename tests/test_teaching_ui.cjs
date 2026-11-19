"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../src/so101_human_follow/static/teaching.js"), "utf8");
const missingMotor = "FeetechMotorsBus motor check failed on port 'COM3':\nMissing motor IDs:\n  - 5 (expected model: 777)\nFull expected motor list (id: model_number):\n{1: 777, 2: 777, 3: 777, 4: 777, 5: 777, 6: 777}";
const flush = () => new Promise(setImmediate);
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function status(overrides = {}) {
  return { robot_allowed: true, state: "FAULT", active: false, recording: false,
    follower_port: "COM4", follower_connected: true, ...overrides };
}

function fixture(initial = status()) {
  class Element {
    constructor(id = "", text = "") {
      this.id = id; this.textContent = text; this.value = ""; this.disabled = false;
      this.checked = false; this.hidden = false; this.open = false; this.dataset = {};
      this.handlers = new Map(); this.children = []; this.adjacent = [];
      this.classList = { toggle() {} };
    }
    addEventListener(event, handler) { this.handlers.set(event, handler); }
    append(...elements) { this.children.push(...elements); }
    insertAdjacentElement(_where, element) { this.adjacent.push(element); }
    insertRow() { const row = new Element(); this.append(row); return row; }
    insertCell() { const cell = new Element(); this.append(cell); return cell; }
    fire(event = "click") { this.handlers.get(event)?.(); }
    click() { if (!this.disabled) this.fire(); }
  }
  const ids = ["teachLeaderPort", "teachFollowerPort", "teachLeaderStatus", "teachFollowerStatus",
    "teachFollowerCalibrationStatus", "teachRecordingMode", "teachRecordingName", "teachRecordingDescription",
    "teachSyncCheck", "teachSyncConfirmed", "teachStartRecording", "teachStopRecording",
    "teachExtractRecording", "teachRecordingStatus", "teachJointRanges", "teachSyncDirections", "baseCalibrationProgressText"];
  const elements = new Map(ids.map(id => [id, new Element(id)]));
  elements.get("teachLeaderPort").value = "COM3";
  elements.get("teachFollowerPort").value = "COM4";
  elements.get("teachRecordingMode").value = "leader";
  const labels = { "calibration/start": "开始主臂校准", "calibration/middle": "确定六轴中位",
    "calibration/range": "采集六轴范围", "calibration/save": "保存主臂校准", "calibration/cancel": "取消主臂校准" };
  const buttons = new Map(Object.entries(labels).map(([operation, label]) => {
    const element = new Element(operation, label); element.dataset.teachCommand = operation;
    return [operation, element];
  }));
  const followerButtons = [];
  const intervals = new Map();
  let nextTimer = 0;
  const shared = { operator: true, robot_connected: true, robot_allowed: true, base_calibration: { active: false } };
  const window = {
    so101FollowerControl: { state: shared, port: "COM4", portDisabled: false },
    addEventListener() {},
    setInterval(handler) { intervals.set(++nextTimer, handler); return nextTimer; },
    clearInterval(timer) { intervals.delete(timer); },
  };
  const context = { window,
    document: { getElementById: id => elements.get(id), createElement: () => new Element(),
      querySelectorAll: selector => selector === "[data-teach-command]" ? [...buttons.values()] : followerButtons },
    Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  const calls = [];
  const handlers = new Map();
  const api = (route, payload) => {
    calls.push({ route, payload });
    return (handlers.get(route) || (() => Promise.resolve(initial)))(payload);
  };
  const panel = window.so101CreateTeachingPanel({
    api, getRecording: () => null, getRecordingPayload: () => ({}), setRecording() {},
    getObservedLimits: () => ({}), getSimulation: () => ({ joints: [] }),
    applyDraft() {}, setReplayBlocked() {}, message() {},
  });
  return { panel, buttons, elements, calls, shared,
    tick: () => { for (const callback of intervals.values()) callback(); },
    on: (route, handler) => handlers.set(route, handler),
    posts: () => calls.filter(call => call.route !== "/api/teach/status"),
    details: () => elements.get("teachLeaderStatus").adjacent[0],
  };
}

test("click immediately shows connection progress; an unsettled POST still blocks close", async () => {
  const f = fixture(); f.panel.start(); await flush();
  const post = deferred();
  f.on("/api/teach/calibration/start", () => post.promise);
  f.buttons.get("calibration/start").click();
  assert.equal(f.buttons.get("calibration/start").textContent, "正在连接…");
  assert.match(f.elements.get("teachLeaderStatus").textContent, /正在连接主臂 COM3/);
  assert.equal(f.buttons.get("calibration/cancel").disabled, true);
  assert.equal(await f.panel.prepareClose(), false);
  assert.equal(f.posts().length, 1);
  post.reject(Object.assign(new Error(missingMotor), { status: 409 })); await flush();
  f.panel.close();
});

test("a rejected POST releases pending even when its status GET hangs, without replaying commands", async () => {
  const f = fixture(); f.panel.start(); await flush();
  const get = deferred();
  f.on("/api/teach/calibration/start", () => Promise.reject(Object.assign(new Error(missingMotor), { status: 409 })));
  f.on("/api/teach/status", () => get.promise);
  f.buttons.get("calibration/start").click(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  assert.equal(f.buttons.get("calibration/start").textContent, "开始主臂校准");
  assert.equal(await f.panel.prepareClose(), true);
  assert.match(f.elements.get("teachLeaderStatus").textContent, /主臂 COM3 未检测到 ID5（腕部旋转），本次校准未开始/);
  assert.match(f.elements.get("teachLeaderStatus").textContent, /同步/);
  assert.equal(f.details().hidden, false);
  assert.equal(f.details().open, false);
  assert.equal(f.details().children[1].textContent, missingMotor);
  const before = f.calls.length;
  f.buttons.get("calibration/start").fire(); // The synchronization gate also protects handlers.
  f.tick(); f.tick(); await flush();
  assert.equal(f.calls.length, before);
  assert.equal(f.posts().length, 1);
  get.resolve(status({ error: missingMotor })); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, false);
  f.buttons.get("calibration/start").click(); await flush();
  assert.equal(f.posts().length, 2);
  f.panel.close();
});

test("status failure keeps new commands gated; a later trusted status restores retry", async () => {
  const f = fixture(); f.panel.start(); await flush();
  f.on("/api/teach/calibration/start", () => Promise.reject(Object.assign(new Error(missingMotor), { status: 409 })));
  f.on("/api/teach/status", () => Promise.reject(new Error("状态读取超时")));
  f.buttons.get("calibration/start").click(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  assert.equal(await f.panel.prepareClose(), true);
  f.buttons.get("calibration/start").fire(); await flush();
  assert.equal(f.posts().length, 1);
  f.on("/api/teach/status", () => Promise.resolve(status())); f.tick(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, false);
  f.panel.close();
});

test("unknown command result permits explicit cancellation; confirmed cancel allows close without GET", async () => {
  const f = fixture(); f.panel.start(); await flush();
  const oldGet = deferred(), cancelGet = deferred(); let reads = 0;
  f.on("/api/teach/status", () => (++reads === 1 ? oldGet.promise : cancelGet.promise));
  f.on("/api/teach/calibration/start", () => Promise.resolve({ ok: true }));
  f.buttons.get("calibration/start").click(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  assert.equal(f.buttons.get("calibration/cancel").disabled, false);
  assert.equal(await f.panel.prepareClose(), false);
  assert.equal(f.posts().length, 1); // No automatic cancel or retry.
  f.on("/api/teach/calibration/cancel", () => Promise.resolve({ ok: true }));
  f.buttons.get("calibration/cancel").click(); await flush();
  assert.equal(f.posts().length, 2);
  assert.equal(await f.panel.prepareClose(), true);
  cancelGet.resolve(status({ state: "IDLE" })); await flush();
  oldGet.resolve(status({ state: "WAITING_MIDDLE", active: true })); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, false);
  assert.equal(f.buttons.get("calibration/cancel").disabled, true);
  assert.equal(await f.panel.prepareClose(), true);
  f.panel.close();
});

test("cancel remains available during status errors for a known calibration; permission gates remain", async () => {
  const f = fixture(status({ state: "WAITING_MIDDLE", active: true })); f.panel.start(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  assert.equal(f.buttons.get("calibration/middle").disabled, false);
  assert.equal(f.buttons.get("calibration/range").disabled, true);
  f.on("/api/teach/status", () => Promise.reject(new Error("状态读取超时"))); f.tick(); await flush();
  assert.equal(f.buttons.get("calibration/middle").disabled, true);
  assert.equal(f.buttons.get("calibration/cancel").disabled, false);
  f.shared.operator = false; f.panel.render();
  assert.equal(f.buttons.get("calibration/cancel").disabled, true);
  assert.equal(f.elements.get("teachStopRecording").disabled, true);
  f.panel.close();
});

test("same-port, preview, other-calibration and fresh-open gates are retained", async () => {
  const f = fixture(); f.panel.start(); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, false);
  f.elements.get("teachLeaderPort").value = "COM4"; f.panel.render();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  f.elements.get("teachLeaderPort").value = "COM3";
  f.shared.base_calibration.active = true; f.panel.render();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  f.shared.base_calibration.active = false; f.panel.close();
  const get = deferred(); f.on("/api/teach/status", () => get.promise); f.panel.start();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  get.resolve(status({ robot_allowed: false })); await flush();
  assert.equal(f.buttons.get("calibration/start").disabled, true);
  assert.equal(f.buttons.get("calibration/cancel").disabled, true);
  assert.equal(f.posts().length, 0);
  f.panel.close();
});
