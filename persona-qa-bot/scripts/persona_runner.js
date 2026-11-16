#!/usr/bin/env node
/**
 * persona_runner.js — drive a live app as many behavioral personas, in
 * parallel, recording the screen and capturing bugs the way real QA does.
 *
 * What makes this "real-world" rather than a scripted click-through:
 *   - Personas have TRAITS (patience, typing speed, typos, misclicks, device,
 *     network) that change HOW they interact — see behaviors.js.
 *   - Passive SENSORS watch every step for console errors, failed network
 *     calls (incl. frontend→DB sync failures), HTTP 4xx/5xx, and layout
 *     overflow — bugs the scripted assertions never check.
 *   - An EXPLORE mode autonomously clicks around (monkey testing) for coverage
 *     without hand-written steps.
 *   - Scale: run personas with --concurrency, and --repeat to surface flaky /
 *     race-condition bugs that only fail some of the time.
 *
 * Playwright + Chromium are preinstalled.
 *
 * Usage:
 *   node persona_runner.js --url https://staging.example.com \
 *     --personas personas/library.json --out ./runs \
 *     --concurrency 4 --repeat 1 --video on
 */
const { chromium, devices } = require("playwright");
const fs = require("fs");
const path = require("path");
const B = require("./behaviors");

function arg(flag, def) {
  const i = process.argv.indexOf(flag);
  return i !== -1 ? process.argv[i + 1] : def;
}

const NETWORK_PROFILES = {
  "slow-3g": { downloadThroughput: (500 * 1024) / 8, uploadThroughput: (500 * 1024) / 8, latency: 400, offline: false },
  "fast-3g": { downloadThroughput: (1.6 * 1024 * 1024) / 8, uploadThroughput: (750 * 1024) / 8, latency: 150, offline: false },
  offline: { downloadThroughput: 0, uploadThroughput: 0, latency: 0, offline: true },
};

// --- step action library --------------------------------------------------
const ACTIONS = {
  async goto(page, s, t, to) {
    await page.goto(s.url, { waitUntil: "load", timeout: to(15000) });
  },
  async click(page, s, t, to) {
    await B.humanClick(page, s.selector, t, to(5000));
  },
  async type(page, s, t, to) {
    await B.humanType(page, s.selector, s.value, t, to(5000));
  },
  async fill(page, s, t, to) {
    await page.fill(s.selector, s.value, { timeout: to(5000) });
  },
  async press(page, s, t, to) {
    await page.keyboard.press(s.key);
  },
  async back(page) {
    await page.goBack({ waitUntil: "load" }).catch(() => {});
  },
  async wait(page, s) {
    await page.waitForTimeout(s.ms ?? 500);
  },
  async expectText(page, s, t, to) {
    const body = await page.textContent("body");
    if (!body || !body.includes(s.text))
      throw new Error(`expected text not found: ${JSON.stringify(s.text)}`);
  },
  async expectVisible(page, s, t, to) {
    await page.waitForSelector(s.selector, { state: "visible", timeout: to(5000) });
  },
};

async function applyNetwork(context, page, profileName) {
  if (!profileName || !NETWORK_PROFILES[profileName]) return;
  const client = await context.newCDPSession(page);
  await client.send("Network.enable").catch(() => {});
  await client.send("Network.emulateNetworkConditions", NETWORK_PROFILES[profileName]).catch(() => {});
}

function newSink() {
  return { consoleErrors: [], pageErrors: [], failedRequests: [], httpErrors: [], layout: [] };
}

// Turn captured signals into bug entries (passive findings).
function signalBugs(sink) {
  const bugs = [];
  for (const e of sink.pageErrors) bugs.push({ kind: "js-error", action: "console", error: e });
  for (const e of sink.consoleErrors) bugs.push({ kind: "console-error", action: "console", error: e });
  for (const r of sink.httpErrors) bugs.push({ kind: "http-error", action: "network", error: `HTTP ${r.status} ${r.url}` });
  for (const r of sink.failedRequests) bugs.push({ kind: "request-failed", action: "network", error: `${r.error} ${r.url}` });
  for (const l of sink.layout)
    if (l.horizontalScroll || l.offCanvasElements > 0)
      bugs.push({ kind: "layout", action: "layout", error: `horizontalScroll=${l.horizontalScroll} offCanvas=${l.offCanvasElements} @ ${l.where}` });
  return bugs;
}

// --- exploratory (monkey) mode -------------------------------------------
async function explore(page, persona, t, sink, runDir, to) {
  const cfg = persona.explore || {};
  await page.goto(cfg.startUrl || "/", { waitUntil: "load", timeout: to(15000) }).catch(() => {});
  const budget = cfg.budget ?? 12;
  const bugs = [];
  for (let i = 0; i < budget; i++) {
    await B.think(page, t);
    await B.maybeScroll(page, t);
    { const L = await B.checkLayout(page); if (L) sink.layout.push({ ...L, where: page.url() }); }
    const handles = await page.$$("a[href], button, [role=button], input[type=submit]");
    if (!handles.length) break;
    const el = handles[Math.floor(Math.random() * handles.length)];
    try {
      await el.click({ timeout: to(3000) });
      await page.waitForLoadState("load", { timeout: to(5000) }).catch(() => {});
    } catch (err) {
      const shot = path.join(runDir, `explore_${i}.png`);
      await page.screenshot({ path: shot }).catch(() => {});
      bugs.push({ kind: "explore-click", step: i, error: String(err.message || err).slice(0, 120), screenshot: shot });
    }
  }
  return bugs;
}

// --- one persona run ------------------------------------------------------
async function runOnce(browser, baseUrl, persona, outDir, opts) {
  const tag = `${persona.name}${opts.iter != null ? `#${opts.iter}` : ""}`.replace(/\s+/g, "_");
  const runDir = path.join(outDir, tag);
  fs.mkdirSync(runDir, { recursive: true });

  const t = B.trait(persona);
  const to = (ms) => Math.round(ms * t.patience); // patience scales timeouts

  const ctxOpts = {
    baseURL: baseUrl,
    ignoreHTTPSErrors: !!process.env.QA_IGNORE_HTTPS_ERRORS,
    ...(persona.device && devices[persona.device] ? devices[persona.device] : { viewport: persona.viewport ?? { width: 1280, height: 800 } }),
  };
  if (opts.video) ctxOpts.recordVideo = { dir: runDir, size: { width: 1280, height: 800 } };

  const context = await browser.newContext(ctxOpts);
  const page = await context.newPage();
  const sink = newSink();
  B.attachSensors(page, sink);
  await applyNetwork(context, page, persona.network);

  const result = { persona: persona.name, iter: opts.iter, goal: persona.goal, device: persona.device || null, network: persona.network || null, bugs: [] };

  try {
    if (persona.mode === "explore") {
      result.bugs.push(...(await explore(page, persona, t, sink, runDir, to)));
    } else {
      for (const [i, step] of (persona.steps || []).entries()) {
        await B.think(page, t);
        await B.maybeScroll(page, t);
        const fn = ACTIONS[step.action];
        if (!fn) {
          result.bugs.push({ kind: "bad-step", step: i, error: `unknown action: ${step.action}` });
          continue;
        }
        try {
          await fn(page, { url: baseUrl, ...step }, t, to);
        } catch (err) {
          const shot = path.join(runDir, `fail_step_${i}.png`);
          await page.screenshot({ path: shot }).catch(() => {});
          result.bugs.push({ kind: "step-failed", step: i, action: step.action, detail: step, error: String(err.message || err).slice(0, 200), screenshot: shot });
        }
        { const L = await B.checkLayout(page); if (L) sink.layout.push({ ...L, where: `step ${i}` }); }
      }
    }
  } finally {
    result.bugs.push(...signalBugs(sink));
    result.signals = { console: sink.consoleErrors.length, pageErrors: sink.pageErrors.length, httpErrors: sink.httpErrors.length, failedRequests: sink.failedRequests.length };
    await context.close(); // flush video
  }

  if (opts.video) {
    const video = fs.readdirSync(runDir).find((f) => f.endsWith(".webm"));
    result.recording = video ? path.join(runDir, video) : null;
  }
  fs.writeFileSync(path.join(runDir, "result.json"), JSON.stringify(result, null, 2));
  return result;
}

// simple concurrency pool
async function pool(items, size, worker) {
  const out = [];
  let i = 0;
  const runners = Array.from({ length: Math.min(size, items.length) }, async () => {
    while (i < items.length) {
      const idx = i++;
      out[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(runners);
  return out;
}

(async () => {
  const baseUrl = arg("--url");
  const personasPath = arg("--personas");
  const outDir = arg("--out", "./runs");
  const concurrency = parseInt(arg("--concurrency", "3"), 10);
  const repeat = parseInt(arg("--repeat", "1"), 10);
  const video = arg("--video", "on") !== "off";
  if (!baseUrl || !personasPath) {
    console.error("usage: persona_runner.js --url <url> --personas <file> [--out dir] [--concurrency N] [--repeat N] [--video on|off]");
    process.exit(1);
  }
  const personas = JSON.parse(fs.readFileSync(personasPath, "utf8"));
  fs.mkdirSync(outDir, { recursive: true });

  // expand personas × repeat into a flat job list (scale + flakiness detection)
  const jobs = [];
  for (const p of personas) for (let r = 0; r < repeat; r++) jobs.push({ persona: p, iter: repeat > 1 ? r : null });

  const proxyServer = process.env.HTTPS_PROXY || process.env.HTTP_PROXY;
  const browser = await chromium.launch({ args: ["--no-sandbox"], ...(proxyServer ? { proxy: { server: proxyServer } } : {}) });

  const t0 = Date.now();
  const results = await pool(jobs, concurrency, (job) => {
    process.stderr.write(`▶ ${job.persona.name}${job.iter != null ? ` #${job.iter}` : ""}\n`);
    return runOnce(browser, baseUrl, job.persona, outDir, { iter: job.iter, video });
  });
  await browser.close();

  // reach = how many distinct persona-runs hit each bug signature.
  const reach = {};
  for (const r of results)
    for (const b of r.bugs) {
      const key = `${b.kind}:${b.action || ""}:${(b.detail && b.detail.selector) || b.error || ""}`.slice(0, 160);
      reach[key] = (reach[key] || 0) + 1;
    }
  // flaky = bug signatures that appear in some but not all repeats of a persona.
  const flaky = {};
  if (repeat > 1) {
    const byPersona = {};
    for (const r of results) (byPersona[r.persona] ||= []).push(r);
    for (const [name, runs] of Object.entries(byPersona)) {
      const seen = {};
      for (const run of runs) for (const b of run.bugs) seen[b.kind + ":" + (b.error || "")] = (seen[b.kind + ":" + (b.error || "")] || 0) + 1;
      for (const [sig, n] of Object.entries(seen)) if (n > 0 && n < runs.length) (flaky[name] ||= []).push({ sig: sig.slice(0, 120), hits: `${n}/${runs.length}` });
    }
  }

  const summary = {
    baseUrl,
    personas: personas.length,
    runs: results.length,
    concurrency,
    repeat,
    durationMs: Date.now() - t0,
    totalBugs: results.reduce((a, r) => a + r.bugs.length, 0),
    reach,
    flaky,
    results,
  };
  fs.writeFileSync(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({ runs: summary.runs, totalBugs: summary.totalBugs, durationMs: summary.durationMs, flakyPersonas: Object.keys(flaky) }, null, 2));
})();
