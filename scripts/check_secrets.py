"""Block secrets and private information before they reach the public repo.

Run by the git hooks (installed with `make hooks`):

    pre-commit   python scripts/check_secrets.py --staged
    pre-push     python scripts/check_secrets.py --push <remote-ref-range>...

It checks the ADDED lines of the change for:
- API keys and tokens (GitHub, Hugging Face, OpenAI-style, AWS, Slack,
  Google), private keys, and values assigned to names like api_key,
  secret, token or password;
- absolute home-folder paths and local tool paths;
- personal email addresses (gmail, yahoo, outlook, hotmail, icloud);
- private IP addresses.

It also checks file names (`.env`, key files; at push time also private
folders and private docs), file sizes (over 10 MB), and images for GPS
data. Extra patterns that must never be published, such as personal
names, go in `.git/info/secret-patterns` (one regular expression per line,
local and never committed), because writing them here would publish them.

Exit code 1 (and a report) if anything is found; 0 otherwise. A finding
that is a false positive can be allowed for one line by adding
`# allow-secret` to that line.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "OpenAI-style key": r"sk-[A-Za-z0-9_-]{20,}",
    "AWS key": r"AKIA[0-9A-Z]{16}",
    "Google API key": r"AIza[0-9A-Za-z_-]{35}",
    "Slack token": r"xox[abprs]-[A-Za-z0-9-]{10,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "assigned secret": r"(?i)(api[_-]?key|apikey|secret|token|passw(or)?d)[\"' ]*[:=][\"' ]*[A-Za-z0-9_\-/+]{16,}",
    "home-folder path": r"/(Users|home)/[A-Za-z][A-Za-z0-9._-]+/",
    "local tool path": r"/private/tmp/|/tmp/claude|\.claude/|/cc-socks/",  # allow-secret (the pattern itself)
    "personal email": r"[A-Za-z0-9._%+-]+@(gmail|googlemail|yahoo|hotmail|outlook|icloud|live|aol|proton(mail)?)\.[a-z.]+",
    "private IP address": r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b",
}
ALWAYS_BLOCKED_FILES = r"(^|/)\.env(\..*)?$|\.(pem|key|p12|pfx)$|(^|/)id_(rsa|ed25519|ecdsa)(\.pub)?$|(^|/)credentials(\.json)?$"
# Private material: blocked at PUSH time only, so local private branches can still commit it.
PRIVATE_FILES = (r"^private/|^docs/PLAN\.md$|^docs/PHASE5|^docs/RELATED_WORK\.md$|^docs/CLUEWORLD|"
                 r"^runs/(?!.*\.gitkeep$)|^data/(?!.*\.gitkeep$)|sardrone/clueworld/")  # fmt: skip  (.gitkeep placeholders are fine)
PRIVATE_BRANCHES = ("phase5-cluemem",)
MAX_BYTES = 10 * 1024 * 1024
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic")
ALLOW = "allow-secret"


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


def local_patterns() -> dict[str, str]:
    path = Path(git("rev-parse", "--git-common-dir").strip()) / "info" / "secret-patterns"
    if not path.exists():
        return {}
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {f"local pattern {k + 1}": ln for k, ln in enumerate(lines)}


def scan_lines(lines: list[tuple[str, int, str]], patterns: dict[str, str]) -> list[str]:
    """(file, line number, text) -> findings. The matched text is shortened, never printed whole."""
    found = []
    compiled = {name: re.compile(p) for name, p in patterns.items()}
    for path, number, text in lines:
        if ALLOW in text:
            continue
        for name, rx in compiled.items():
            for piece in windows(text):
                m = rx.search(piece)
                if m:
                    hit = m.group(0)
                    shown = hit[:4] + "..." if len(hit) > 8 else "..."
                    found.append(f"{path}:{number}: {name} ({shown})")
                    break
    return found


WINDOW, OVERLAP = 4096, 256


def windows(text: str):
    """Very long lines (minified code, data) are scanned in overlapping
    windows: some patterns backtrack, and on a single line of millions of
    characters they would take effectively forever."""
    if len(text) <= WINDOW:
        yield text
        return
    for start in range(0, len(text), WINDOW - OVERLAP):
        yield text[start:start + WINDOW]


def added_lines(diff: str) -> list[tuple[str, int, str]]:
    """Added lines from a unified diff (git diff -U0), with their file and new line number."""
    out, path, number = [], None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            number = int(m.group(1)) if m else 0
        elif line.startswith("+") and path:
            out.append((path, number, line[1:]))
            number += 1
    return out


def image_has_gps(data: bytes) -> bool:
    try:
        from PIL import Image

        return 34853 in Image.open(io.BytesIO(data)).getexif()  # 34853 = GPSInfo
    except Exception:  # noqa: BLE001 - not an image Pillow can read: nothing to check
        return False


def check_files(files: list[str], read, private_too: bool) -> list[str]:
    found = []
    for f in files:
        if re.search(ALWAYS_BLOCKED_FILES, f):
            found.append(f"{f}: file type that holds secrets")
        if private_too and re.search(PRIVATE_FILES, f):
            found.append(f"{f}: private material (must not be pushed to the public repo)")
        data = read(f)
        if data is None:
            continue
        if len(data) > MAX_BYTES:
            found.append(f"{f}: {len(data) / 1e6:.1f} MB (over 10 MB)")
        if f.lower().endswith(IMAGE_SUFFIXES) and image_has_gps(data):
            found.append(f"{f}: image with GPS location data")
    return found


def check_staged() -> list[str]:
    patterns = {**PATTERNS, **local_patterns()}
    findings = scan_lines(added_lines(git("diff", "--cached", "-U0", "--no-color")), patterns)
    files = [f for f in git("diff", "--cached", "--name-only", "--diff-filter=AM").splitlines() if f]

    def read(f):
        try:
            return git("show", f":{f}", binary=True)
        except subprocess.CalledProcessError:
            return None

    return findings + check_files(files, read, private_too=False)


def check_push(ranges: list[str], branches: list[str]) -> list[str]:
    findings = [f"branch {b}: private branch, never pushed" for b in branches if b in PRIVATE_BRANCHES]
    patterns = {**PATTERNS, **local_patterns()}
    for rng in ranges:
        findings += scan_lines(added_lines(git("log", "-p", "-U0", "--no-color", "--format=", rng)), patterns)
        files = sorted({f for f in git("log", "--name-only", "--format=", "--diff-filter=AM", rng).splitlines() if f})
        tip = rng.split("..")[-1]

        def read(f, tip=tip):
            try:
                return git("show", f"{tip}:{f}", binary=True)
            except subprocess.CalledProcessError:
                return None  # deleted again later in the range

        findings += check_files(files, read, private_too=True)
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Block secrets and private information before commit and push.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="check the staged changes (pre-commit)")
    group.add_argument("--push", nargs="*", metavar="RANGE", help="check these commit ranges (pre-push)")
    parser.add_argument("--branch", action="append", default=[], help="branch being pushed (pre-push)")
    args = parser.parse_args(argv)
    findings = check_staged() if args.staged else check_push(args.push or [], args.branch)
    if findings:
        print("Blocked: possible secret or private information:", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        print("Fix it, or add '# allow-secret' to a line that is a false positive.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
