"""Shared experiment plumbing: seeds, output paths, CLI helpers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Published results live in results/. --quick smoke tests write to results_quick/ so that they never overwrite
# them; run_all.sh --quick points every script there through FISHER_SURROGATE_RESULTS.
RESULTS_DIR = Path(os.environ.get("FISHER_SURROGATE_RESULTS") or Path(__file__).resolve().parents[1] / "results")
QUICK_DIR = RESULTS_DIR.parent / "results_quick"

# Seed conventions inherited from the notebooks so that runs are reproducible.
BASE_SEED = 2026
RESIDUAL_SEED_OFFSET = 5000
PI_SEED_OFFSET = 9000
DEEPONET_SEED_OFFSET = 12000
NOISE_SEED_OFFSET = 10000
OOD_DATA_SEED_BASE = 9000


def experiment_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="output directory (default: results/<name>)")
    p.add_argument("--device", default=None)
    return p


def out_dir(args, name):
    d = args.out or ((QUICK_DIR if getattr(args, "quick", False) else RESULTS_DIR) / name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def final_tables_dir(quick=False):
    """tables/final/: manuscript tables in their final layout, written by the scripts that produce them.
    Smoke tests (--quick, or run_all.sh --quick) write to results_quick/tables_final/ instead."""
    default = Path(__file__).resolve().parents[1] / "results"
    d = QUICK_DIR / "tables_final" if (quick or RESULTS_DIR != default) else RESULTS_DIR.parent / "tables" / "final"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(out, name, seed_df=None, summary=None, latex=None, config=None):
    if seed_df is not None:
        seed_df.to_csv(out / f"{name}_seed_results.csv", index=False)
    if summary is not None:
        summary.to_csv(out / f"{name}_summary.csv", index=False)
    if latex is not None:
        (out / f"{name}_table.tex").write_text(latex, encoding="utf-8")
    if config is not None:
        (out / f"{name}_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"saved to {out}")
    if summary is not None:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary)
