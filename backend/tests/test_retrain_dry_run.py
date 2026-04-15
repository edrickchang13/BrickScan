"""
CI wrapper for the retrain dry-run smoke test.

Runs ml/scripts/smoke_retrain_dry_run.sh and asserts exit 0 + expected
success-log line. Catches drift between retrain_from_feedback.py and the
feedback CSV schema before it hits DGX Spark.

Requires: pandas + torch in the active Python environment. Skipped if either
is missing (CI environments without the ML extras installed).
"""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_SCRIPT = REPO_ROOT / "ml" / "scripts" / "smoke_retrain_dry_run.sh"


def _missing_ml_deps():
    try:
        import pandas  # noqa: F401
        import torch   # noqa: F401
        return None
    except ImportError as e:
        return str(e)


@pytest.mark.skipif(
    not SMOKE_SCRIPT.exists(),
    reason="smoke_retrain_dry_run.sh not present",
)
@pytest.mark.skipif(
    _missing_ml_deps() is not None,
    reason=f"ML deps missing: {_missing_ml_deps()}",
)
def test_retrain_dry_run_passes():
    """The dry-run smoke script exits 0 and prints the success banner."""
    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert result.returncode == 0, (
        f"smoke script exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Dry run OK" in result.stdout, (
        "Expected 'Dry run OK' in output.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Dry run PASSED" in result.stdout, (
        "Expected 'Dry run PASSED' in output.\n"
        f"stdout:\n{result.stdout}"
    )
