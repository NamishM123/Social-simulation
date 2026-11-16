/**
 * behaviors.js — human-like interaction + passive signal capture.
 *
 * Real users don't execute clean step lists. They type slowly and make typos,
 * misclick, lose patience and double-submit, scroll, hit back. And real bugs
 * mostly surface as console errors, failed network calls (frontend→DB sync!),
 * and layout overflow — not as "element not found". This module gives the
 * runner both: trait-driven actions, and sensors that watch every step.
 */

// Trait defaults. A persona overrides any of these; everything stays a plain
// number/bool so personas are pure data (JSON).
const TRAIT_DEFAULTS = {
  patience: 1.0, // multiplies timeouts: 0.3 = gives up fast, 2 = very patient
  typingDelayMs: 0, // per-character delay when typing
  typoRate: 0, // chance per char of typing a wrong key then correcting
  misclickRate: 0, // chance a click first lands near-but-off, then corrects
  doubleSubmit: false, // impatient users click submit again before it responds
  scrolliness: 0, // chance to scroll the page between steps
  thinkMsRange: [0, 0], // random pause between steps (reading/deciding)
};

function trait(persona) {
  return { ...TRAIT_DEFAULTS, ...(persona.traits || {}) };
}

const rand = (a, b) => a + Math.random() * (b - a);
const chance = (p) => Math.random() < p;

// --- passive sensors ------------------------------------------------------
// Attach once per page; everything collects into `sink`. These catch bugs the
// scripted steps never assert on.
function attachSensors(page, sink) {
  page.on("console", (m) => {
    if (m.type() === "error") sink.consoleErrors.push(m.text().slice(0, 300));
  });
  page.on("pageerror", (e) => sink.pageErrors.push(String(e.message || e).slice(0, 300)));
  page.on("requestfailed", (r) => {
    const f = r.failure();
    sink.failedRequests.push({ url: r.url(), error: f && f.errorText });
  });
  page.on("response", (r) => {
    const s = r.status();
    if (s >= 400) sink.httpErrors.push({ url: r.url().slice(0, 200), status: s });
  });
}

// Layout / UI-breaking check: horizontal overflow and off-canvas elements.
// Returns null if the page is mid-navigation (context destroyed) rather than
// throwing — the caller treats null as "not measured".
async function checkLayout(page) {
  try {
    return await page.evaluate(() => {
    const docW = document.documentElement.clientWidth;
    let off = 0;
    for (const el of document.querySelectorAll("body *")) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.right > docW + 2) off++;
    }
      return {
        horizontalScroll: document.documentElement.scrollWidth > docW + 2,
        offCanvasElements: off,
      };
    });
  } catch {
    return null; // navigation in flight; nothing to measure
  }
}

// --- human-like actions ---------------------------------------------------
async function think(page, t) {
  const [lo, hi] = t.thinkMsRange;
  if (hi > 0) await page.waitForTimeout(rand(lo, hi));
}

async function maybeScroll(page, t) {
  if (chance(t.scrolliness)) {
    await page.mouse.wheel(0, rand(200, 900)).catch(() => {});
    await page.waitForTimeout(150);
  }
}

async function humanType(page, selector, text, t, timeout) {
  await page.click(selector, { timeout });
  for (const ch of text) {
    if (chance(t.typoRate)) {
      const wrong = String.fromCharCode(97 + Math.floor(Math.random() * 26));
      await page.keyboard.type(wrong, { delay: t.typingDelayMs });
      await page.keyboard.press("Backspace");
    }
    await page.keyboard.type(ch, { delay: t.typingDelayMs });
  }
}

async function humanClick(page, selector, t, timeout) {
  const el = await page.waitForSelector(selector, { timeout, state: "visible" });
  if (chance(t.misclickRate)) {
    const box = await el.boundingBox();
    if (box) {
      // land just outside the element first, then correct — catches overlay/
      // z-index bugs where a near-miss hits the wrong thing.
      await page.mouse.click(box.x - 4, box.y - 4).catch(() => {});
      await page.waitForTimeout(120);
    }
  }
  await el.click({ timeout });
  if (t.doubleSubmit) {
    await el.click({ timeout, force: true }).catch(() => {});
  }
}

module.exports = {
  trait,
  attachSensors,
  checkLayout,
  think,
  maybeScroll,
  humanType,
  humanClick,
  rand,
  chance,
};
