"""Both eval scripts accept --questions and fail clearly on a missing file."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(module: str):
    return subprocess.run(
        [sys.executable, "-m", module, "--questions", "/nonexistent/q.jsonl"],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_validate_eval_reports_a_missing_questions_file():
    r = _run("scripts.validate_eval")
    assert r.returncode != 0
    assert "questions file not found" in r.stdout + r.stderr


def test_eval_retrieval_reports_a_missing_questions_file():
    r = _run("scripts.eval_retrieval")
    assert r.returncode != 0
    assert "questions file not found" in r.stdout + r.stderr