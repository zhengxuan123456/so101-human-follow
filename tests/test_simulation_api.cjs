"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise the actual private request helper without opening a browser,
// starting the server, or sending any hardware requests.
const source = fs.readFileSync(path.join(__dirname, "../src/so101_human_follow/static/simulation.js"), "utf8");
const start = source.indexOf("  async function api(");
const end = source.indexOf("\n  function message(", start);
assert.ok(start >= 0 && end > start, "simulation request helper is present");
const requestSource = source.slice(start, end);

function fixture(fetchImplementation) {
  const timers = new Map();
  const parent = new AbortController();
  const calls = [];
  let nextTimer = 0;
  const context = {
    AbortController,
    controller: parent,
    headers: () => ({ "X-Client-Id": "offline-read-test" }),
    fetch: (url, options) => {
      calls.push({ url, options });
      return fetchImplementation(url, options);
    },
    window: {
      setTimeout(callback, duration) {
        assert.equal(duration, 5000);
        timers.set(++nextTimer, callback);
        return nextTimer;
      },
      clearTimeout(id) { timers.delete(id); },
    },
  };
  const api = vm.runInNewContext(`${requestSource}\napi`, context);
  return {
    api, parent, calls, timers,
    expire() {
      assert.equal(timers.size, 1);
      const [id, callback] = timers.entries().next().value;
      timers.delete(id);
      callback();
    },
  };
}

function waitForAbort(signal) {
  return new Promise((_, reject) => {
    const abort = () => reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });
}

test("a stalled status GET times out and never resends the request", async () => {
  const f = fixture((_, { signal }) => waitForAbort(signal));
  const pending = f.api("/api/teach/status");
  assert.equal(f.calls[0].options.method, "GET");
  f.expire();
  await assert.rejects(pending, error => error.code === "READ_TIMEOUT");
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].options.signal.aborted, true);
  assert.equal(f.parent.signal.aborted, false);
  assert.equal(f.timers.size, 0);
});

test("the deadline also covers a stalled JSON response body", async () => {
  const f = fixture(async (_, { signal }) => ({ ok: true, json: () => waitForAbort(signal) }));
  const pending = f.api("/api/teach/status");
  await Promise.resolve();
  await Promise.resolve();
  f.expire();
  await assert.rejects(pending, error => error.code === "READ_TIMEOUT");
  assert.equal(f.timers.size, 0);
});

test("closing the dialog still aborts a GET immediately", async () => {
  const f = fixture((_, { signal }) => waitForAbort(signal));
  const pending = f.api("/api/teach/status");
  f.parent.abort();
  await assert.rejects(pending, error => error.name === "AbortError" && error.code !== "READ_TIMEOUT");
  assert.equal(f.timers.size, 0);
});

test("a hardware POST gets no read deadline or automatic retry", async () => {
  const f = fixture((_, { signal }) => waitForAbort(signal));
  const pending = f.api("/api/teach/calibration/start", { port: "COM3" });
  assert.equal(f.calls[0].options.method, "POST");
  assert.equal(f.calls[0].options.signal, f.parent.signal);
  assert.equal(f.calls[0].options.body, JSON.stringify({ port: "COM3" }));
  assert.equal(f.timers.size, 0);
  await Promise.resolve();
  assert.equal(f.calls.length, 1);
  f.parent.abort();
  await assert.rejects(pending, error => error.name === "AbortError");
});

test("a server rejection retains its status and hardware error", async () => {
  const reason = "Missing motor IDs: 5";
  const f = fixture(async () => ({ ok: false, status: 409, json: async () => ({ error: reason }) }));
  await assert.rejects(f.api("/api/teach/status"), error => error.status === 409 && error.message === reason);
  assert.equal(f.timers.size, 0);
});

test("successful reads remove the parent's abort listener", async () => {
  const f = fixture(async () => ({ ok: true, json: async () => ({ state: "IDLE" }) }));
  let listeners = 0;
  const add = f.parent.signal.addEventListener.bind(f.parent.signal);
  const remove = f.parent.signal.removeEventListener.bind(f.parent.signal);
  f.parent.signal.addEventListener = (...args) => { listeners += 1; return add(...args); };
  f.parent.signal.removeEventListener = (...args) => { listeners -= 1; return remove(...args); };
  const value = await f.api("/api/teach/status");
  assert.equal(value.state, "IDLE");
  assert.equal(listeners, 0);
  assert.equal(f.timers.size, 0);
});
