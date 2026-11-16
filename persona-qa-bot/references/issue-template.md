# GitHub issue template

The only output format the team wants. No report, no dashboard — a concise
GitHub issue that drops straight into their workflow, with the recording
attached. `file_issue.py` produces exactly this.

```markdown
**Persona:** <which persona hit it>
**Priority:** <P0/P1/P2>  (severity: <how bad for the user> × reach: <how many personas hit it>)

## Screen recording

[▶ Download / view recording](<url>)
<video src="<url>" controls></video>

## Steps to reproduce

1. ...
2. ...
3. ...

## Root cause

<the actual cause, not just the symptom>

## Recommended fix

<concrete change to make>
```

## Priority rule

`priority = severity × reach`

- **severity** — how badly it hurts a user who hits it (blocks the core task? cosmetic?)
- **reach** — how many personas hit it (the runner counts this in `summary.json` → `reach`)

A bug every persona hits outranks an edge-case one, even at equal severity.
