"""demo.py runs on the bundled sample and reproduces the committed numbers."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_runs_on_bundled_sample(tmp_path):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "demo.py"), "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "ONE DECISION, TAKEN APART" in text
    # D3QN alone on the 4 bundled hours, same numbers as the full-split replay's episodes 11-14.
    for pnl in ("+308.41", "-169.21", "+8.62", "-328.78"):
        assert pnl in text
    assert (tmp_path / "steps.csv").is_file()
