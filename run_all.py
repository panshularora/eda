"""
One-command reproducible workflow for Data Vortex Round 1, Phase 1 (Team SE7EN).

    pip install -r requirements.txt
    python run_all.py

Runs, in order, stopping at the first failure:
  1. src/clean.py         raw CSVs  -> cleaned CSVs + audit + change log
  2. src/validate.py      contract checks + byte-for-byte reproducibility
  3. src/export_json.py   single-file JSON copy of both cleaned tables (round-trip checked)
  4. src/eda.py           statistics + figures
  5. src/build_sqlite.py  SQLite database + SQL-vs-pandas cross-checks
  6. src/build_report.py  HTML/PDF report assembled from the outputs above
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["clean.py", "validate.py", "export_json.py", "eda.py", "build_sqlite.py", "build_report.py"]

if __name__ == "__main__":
    start = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for i, step in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] {step} ===", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "src" / step)], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"\nStep {step} failed (exit {result.returncode}). Stopping.")
    print(f"\nAll {len(STEPS)} steps completed in {time.time() - start:.0f}s.")
