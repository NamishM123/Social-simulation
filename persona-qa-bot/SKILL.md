---
name: persona-qa-bot
description: >-
  Test a running web or mobile app the way real users would, using multiple
  personas, BEFORE any real user hits the bug. Use when asked to QA an app,
  find regressions, check core flows (login/auth, data sync, UI), or turn
  found bugs into GitHub issues. Output is GitHub issues + screen recordings,
  never a dashboard or a long report.
---

# Persona QA Bot

Exercise an app end-to-end as a set of distinct user personas, catch bugs
before launch, and file each as a reproducible GitHub issue with a screen
recording attached. This fills the gap left by AI-assisted dev: teams ship
faster, review their own code less, and don't do exhaustive or full-workflow
testing — so hidden bugs slip through.

## When to use

- QA a staging/preview build before release
- Catch regressions between versions
- Verify a specific flow (login, checkout, data save) still works
- Convert observed failures into dev-ready GitHub issues

## Workflow

1. **Pick personas.** Start from `personas/library.json` (10 behavioral
   personas) or add your own — see `references/persona-template.md`. Personas
   are data: a goal, behavioral **traits**, an optional **device** and
   **network** profile, and either scripted `steps` or `mode: "explore"`.
2. **Run them against the live app**, in parallel, recording each:
   ```bash
   node scripts/persona_runner.js --url https://staging.example.com \
     --personas personas/library.json --out ./runs \
     --concurrency 4 --repeat 3 --video on
   ```
   - `--concurrency N` runs N personas at once (scale).
   - `--repeat N` runs each persona N times to surface **flaky / race bugs**
     (a signature that fails some-but-not-all runs is flagged in `summary.json`
     under `flaky`).
   Each run gets `runs/<Persona>/` with a `.webm`, failure screenshots, and
   `result.json`. `runs/summary.json` holds `reach` (how many runs hit each
   bug) and `flaky`.
3. **Triage.** For each candidate bug, write steps to reproduce, root cause,
   and a recommended fix. Score priority as `severity × reach`.
4. **File it as a GitHub issue** with the recording attached:
   ```bash
   python3 scripts/file_issue.py --repo owner/name \
     --title "Login button does nothing on mobile Safari" \
     --persona "New mobile user" --priority P1 \
     --severity "blocks signup" --reach "3 of 4 personas" \
     --steps steps.md --root-cause "handler unbound after hydration" \
     --fix "Bind handler in useEffect" --recording runs/New_mobile_user/*.webm
   ```

## What to test (priority order)

1. **Core functionality** — does the main thing the app does actually work?
2. **Login & authentication flows**
3. **Regressions** — run the same personas against each release
4. **Frontend → DB sync** — does a user action actually save to the backend?
   (save → reload → assert it persisted)
5. **Security flaws**
6. **UI breaking** — web breaks easily on change; mobile breaks when frontend
   and backend fall out of sync

## Priority scoring

`priority = severity (how bad for the user) × reach (how many personas hit it)`

Multi-persona runs give reach for free and `persona_runner.js` counts it in
`summary.json`. A bug that trips every persona ranks above an edge-case one.

## Output format (important)

**No reports. No dashboard.** Conciseness is the whole point. Deliver results
as **GitHub issues** (`references/issue-template.md`) so they drop straight
into the team's workflow. Each issue: screen recording + steps to reproduce +
root cause + priority + recommended fix.

## Positioning

Like FullStory, but *before* launch. FullStory replays real users after they
hit bugs; this catches issues before any real user does.

## How it mimics real users (not scripted click-throughs)

- **Traits** (`behaviors.js`) change *how* a persona acts: `patience` (scales
  timeouts / gives up early), `typingDelayMs`, `typoRate`, `misclickRate`,
  `doubleSubmit` (impatient re-clicks), `scrolliness`, `thinkMsRange`.
- **Devices & networks** — any Playwright device (e.g. `"iPhone 13"`) and
  throttling profiles (`slow-3g`, `fast-3g`, `offline`) per persona.
- **Passive sensors** fire on *every* step, catching bugs no assertion checks:
  JS/page errors, console errors, HTTP 4xx/5xx, failed requests (this is how
  **frontend→DB sync failures** show up), and **layout overflow** (UI breaking).
- **Explore mode** (`mode: "explore"`) autonomously clicks around for coverage
  without hand-written steps (monkey testing).

## Files

- `scripts/persona_runner.js` — parallel harness: traits, devices, networks,
  sensors, explore mode, repeat/flaky detection
- `scripts/behaviors.js` — human-like actions + passive signal capture
- `scripts/file_issue.py` — files a GitHub issue (via `gh`) with the recording
- `personas/library.json` — 10 ready-made behavioral personas
- `references/persona-template.md` — persona JSON shape + coverage checklist
- `references/issue-template.md` — the required issue format + priority rule

## Open scope questions

- Security testing — in scope?
- Frontend → DB sync testing — in scope?
