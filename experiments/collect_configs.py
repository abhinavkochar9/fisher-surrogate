"""Collect every per-experiment *_config.json into results/CONFIG_ALL.json: one file holding every
hyper-parameter, seed, solver tolerance and data setting behind the published results.

Run:  python experiments/collect_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fisher_surrogate.common import RESULTS_DIR  # noqa: E402


def main():
    out = {f.relative_to(RESULTS_DIR).as_posix(): json.loads(f.read_text())
           for f in sorted(RESULTS_DIR.rglob("*_config.json"))}
    (RESULTS_DIR / "CONFIG_ALL.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"{len(out)} experiment configs -> {RESULTS_DIR / 'CONFIG_ALL.json'}")


if __name__ == "__main__":
    main()
