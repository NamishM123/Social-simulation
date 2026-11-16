#!/usr/bin/env python3
"""File a QA bug as a GitHub issue with a screen recording attached.

The team's required output format: a GitHub issue carrying steps to reproduce,
root cause, priority, and a recommended fix, with the recording attached so a
dev can see and reproduce the bug themselves.

GitHub's REST API cannot upload a video to an issue body, so the recording is
attached via a release asset (durable public URL) and embedded in the issue.
Requires the `gh` CLI, already authenticated.

Usage:
  file_issue.py \
      --repo owner/name \
      --title "Login button does nothing on mobile Safari" \
      --persona "Mobile new user" \
      --priority P1 \
      --severity "blocks signup" \
      --reach "3 of 4 personas" \
      --steps steps.md \
      --root-cause "onClick handler not bound after hydration" \
      --fix "Bind handler in useEffect / guard against null ref" \
      --recording run-01.webm
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def sh(*args: str) -> str:
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"command failed: {' '.join(args)}\n{out.stderr}")
    return out.stdout.strip()


def read(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    return p.read_text() if p.exists() else path  # allow inline text too


def upload_recording(repo: str, recording: str) -> str:
    """Attach the recording as a release asset and return its URL."""
    if not recording:
        return ""
    rec = Path(recording)
    if not rec.exists():
        sys.exit(f"recording not found: {recording}")
    tag = "qa-recordings"
    # Ensure a holding release exists (idempotent).
    existing = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True, text=True,
    )
    if existing.returncode != 0:
        sh("gh", "release", "create", tag, "--repo", repo,
           "--title", "QA bot recordings", "--notes",
           "Screen recordings attached to QA issues. Auto-managed.")
    sh("gh", "release", "upload", tag, str(rec), "--repo", repo, "--clobber")
    owner_name = repo
    return (f"https://github.com/{owner_name}/releases/download/"
            f"{tag}/{rec.name}")


def build_body(a: argparse.Namespace, recording_url: str) -> str:
    rec_block = ""
    if recording_url:
        rec_block = (
            f"\n## Screen recording\n\n"
            f"[▶ Download / view recording]({recording_url})\n\n"
            f"<video src=\"{recording_url}\" controls></video>\n"
        )
    return f"""**Persona:** {a.persona}
**Priority:** {a.priority}  (severity: {a.severity} × reach: {a.reach})
{rec_block}
## Steps to reproduce

{read(a.steps)}

## Root cause

{read(a.root_cause)}

## Recommended fix

{read(a.fix)}

---
*Filed by persona-qa-bot before launch.*
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--title", required=True)
    p.add_argument("--persona", required=True)
    p.add_argument("--priority", required=True, help="e.g. P0/P1/P2")
    p.add_argument("--severity", required=True, help="how bad for the user")
    p.add_argument("--reach", required=True, help="how many personas hit it")
    p.add_argument("--steps", required=True, help="file path or inline text")
    p.add_argument("--root-cause", dest="root_cause", required=True)
    p.add_argument("--fix", required=True, help="file path or inline text")
    p.add_argument("--recording", help="path to .webm/.mp4 screen recording")
    p.add_argument("--label", action="append", default=["qa-bot", "bug"])
    a = p.parse_args()

    recording_url = upload_recording(a.repo, a.recording) if a.recording else ""
    body = build_body(a, recording_url)

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        body_file = f.name

    url = sh("gh", "issue", "create", "--repo", a.repo,
             "--title", a.title, "--body-file", body_file,
             *sum([["--label", l] for l in a.label], []))
    print(url)


if __name__ == "__main__":
    main()
