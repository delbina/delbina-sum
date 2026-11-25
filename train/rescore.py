# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Re-score a saved eval_test.jsonl (from the HF model repo) with the current metrics.

    python train/rescore.py path/to/eval_test.jsonl            # v0 outputs
    python train/rescore.py path/to/eval_test.jsonl --data v1
    python train/rescore.py path/to/eval_test.jsonl --data v2       # v2, v3 and v4 share v2's test split
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]
sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))  # evaluate() needs no torch

import sft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--data", default="v0", choices=["v0", "v1", "v2"])
    a = ap.parse_args()
    md = __import__({"v0": "make_dataset", "v1": "make_dataset_v1", "v2": "make_dataset_v2"}[a.data])
    recs = [json.loads(l) for l in open(a.path, encoding="utf-8")]
    rows = [md.make_sample("test", int(r["id"].rsplit("-", 1)[1])) for r in recs]
    assert all(r["id"] == s["id"] for r, s in zip(recs, rows)), "ids don't match the regenerated split"
    for who in ("base", "tuned"):
        if who not in recs[0]:  # eval_test_v0.jsonl has tuned only
            continue
        res = sft.evaluate(rows, [{"summary": r[who], "truncated": False} for r in recs], md.TEST_KINDS)
        res.pop("recall_by_where")
        print(who, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
