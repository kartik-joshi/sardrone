"""The secret check run by the git hooks (scripts/check_secrets.py).

Each test builds a throwaway git repository. Fake secrets are assembled from
pieces at run time, so this file itself holds nothing the hook would block."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"
spec = importlib.util.spec_from_file_location("check_secrets", SCRIPT)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

FAKE_GITHUB = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
FAKE_HOME = "/" + "Users" + "/someone/" + "project/file.txt"
FAKE_EMAIL = "someone" + "@" + "gmail" + ".com"


def git(repo, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout  # fmt: skip


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.org")
    (tmp_path / "README.md").write_text("hello\n")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-q", "-m", "start")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def stage(repo, name, text):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(repo, "add", name)


@pytest.mark.parametrize("text, kind", [
    (f"token = '{FAKE_GITHUB}'\n", "GitHub token"),
    (f"open('{FAKE_HOME}')\n", "home-folder path"),
    (f"contact {FAKE_EMAIL}\n", "personal email"),
    ("api_key = " + "Zx9" * 8 + "\n", "assigned secret"),
    ("-----BEGIN " + "RSA PRIVATE KEY-----\n", "private key"),
    ("server 192.168" + ".1.20\n", "private IP address"),
])  # fmt: skip
def test_staged_secrets_are_blocked(repo, text, kind):
    stage(repo, "code.py", text)
    findings = cs.check_staged()
    assert any(kind in f for f in findings), findings
    assert all(FAKE_GITHUB not in f for f in findings)  # the report never prints a whole secret


def test_clean_changes_pass_and_allow_marker_works(repo):
    stage(repo, "code.py", "x = 1\nprint('hello')\n")
    assert cs.check_staged() == []
    stage(repo, "code.py", f"pattern = r'{FAKE_HOME}'  # allow-secret (an example)\n")
    assert cs.check_staged() == []


def test_local_patterns_catch_personal_names(repo):
    Path(git(repo, "rev-parse", "--git-common-dir").strip(), "info").mkdir(exist_ok=True)
    Path(git(repo, "rev-parse", "--git-common-dir").strip(), "info", "secret-patterns").write_text("(?i)jane\\s+doe\n")
    stage(repo, "notes.md", "Thanks to Jane Doe for the data.\n")
    assert any("local pattern" in f for f in cs.check_staged())


def test_secret_files_and_big_files_are_blocked(repo):
    stage(repo, ".env", "X=1\n")
    stage(repo, "big.bin", ("0" * 999 + "\n") * (cs.MAX_BYTES // 1000 + 1))  # many lines: realistic, and fast to scan
    findings = cs.check_staged()
    assert any(".env" in f for f in findings) and any("over 10 MB" in f for f in findings)


def test_push_blocks_private_files_and_private_branches(repo):
    base = git(repo, "rev-parse", "HEAD").strip()
    stage(repo, "private/notes.md", "draft\n")
    git(repo, "commit", "-q", "-m", "private")
    tip = git(repo, "rev-parse", "HEAD").strip()
    findings = cs.check_push([f"{base}..{tip}"], ["phase5-cluemem"])
    assert any("private material" in f for f in findings)
    assert any("private branch" in f for f in findings)


def test_push_of_clean_commits_passes(repo):
    base = git(repo, "rev-parse", "HEAD").strip()
    stage(repo, "docs/guide.md", "How to run.\n")
    git(repo, "commit", "-q", "-m", "docs")
    tip = git(repo, "rev-parse", "HEAD").strip()
    assert cs.check_push([f"{base}..{tip}"], ["main"]) == []


def test_images_with_gps_are_blocked(repo):
    from PIL import Image

    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif[34853] = {1: "N", 2: (40.0, 0.0, 0.0)}
    img.save(repo / "photo.jpg", exif=exif)
    git(repo, "add", "photo.jpg")
    assert any("GPS" in f for f in cs.check_staged())


def test_a_very_long_line_is_scanned_quickly_and_still_checked():
    import time

    line = "0" * 2_000_000 + FAKE_EMAIL + "0" * 1000
    start = time.perf_counter()
    findings = cs.scan_lines([("data.txt", 1, line)], cs.PATTERNS)
    assert time.perf_counter() - start < 10  # it used to backtrack for minutes
    assert any("personal email" in f for f in findings)
