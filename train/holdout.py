# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Final ship check on held-out chats nobody has looked at.

The test splits (v0 test, v2 test) are reused every run, and each new data version was
shaped by reading that run's failures on them. So a pass there partly means "fits these
300 chats". The holdout split is drawn from the same test distribution (test projects,
held-out kinds and phrasings) with different seeds, and is used once per candidate:

    python train/holdout.py --revision v4             # merged model from delbina/delbina-sum@v4
    python train/holdout.py --revision v4 --base      # also score plain Qwen for the recall-gain bar

Rules, so it stays held out:
  - run it only on a model that already passes all 6 bars on the test splits
  - read only the aggregate lines this prints; never open the saved jsonl to design datagen,
    and never add a diagnose mode for it. If you do, it is just a third test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
REPO = "delbina/delbina-sum"
# keep in sync with notebooks/train.ipynb (settings cell)
BAR, INVENTED_MAX, UNSEEN_MIN, DISTRACTOR_MAX = 15.0, 2.0, 70.0, 5.0
V0_UNSEEN_MIN, V0_INVENTED_MAX = 60.0, 2.0
SHOW = ("fact_recall", "unseen_recall", "invented_rate", "distractor_hit_rate", "stale_hit_rate", "truncated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True, help="branch of delbina/delbina-sum, e.g. v4")
    ap.add_argument("--n", type=int, default=300, help="holdout chats per set")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", action="store_true", help="also run plain Qwen (needed for the recall-gain bar)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "holdout"))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import make_dataset as md0
    import make_dataset_v2 as md2  # v3/v4 test and holdout are v2's
    import sft

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    dev = "cuda" if cuda else "cpu"
    print("device:", torch.cuda.get_device_name(0) if cuda else "cpu (slow: expect a long run)", flush=True)

    sets = {"v2": [md2.make_sample("holdout", i) for i in range(a.n)],
            "v0": [md0.make_sample("holdout", i) for i in range(a.n)]}
    unseen = md2.TEST_KINDS | md0.TEST_KINDS
    tok = AutoTokenizer.from_pretrained(BASE)

    def run(model_id: str, revision: str | None, name: str) -> dict:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, device_map=dev)
        model.config.use_cache = True
        res, outs = {}, {}
        for k, rows in sets.items():
            t = time.time()
            outs[k] = sft.summarize(model, tok, rows, batch_size=a.batch)
            res[k] = sft.evaluate(rows, outs[k], unseen)
            print(f"{name} holdout-{k}: {time.time() - t:.0f}s  " + "  ".join(f"{m} {res[k].get(m)}" for m in SHOW),
                  flush=True)
        del model
        if cuda:
            torch.cuda.empty_cache()
        return {"res": res, "outs": outs}

    tuned = run(REPO, a.revision, "tuned")
    base = run(BASE, None, "base") if a.base else None

    t2, t0 = tuned["res"]["v2"], tuned["res"]["v0"]
    checks = {f"invented <= {INVENTED_MAX}%": t2["invented_rate"] <= INVENTED_MAX,
              f"unseen-kind recall >= {UNSEEN_MIN}": (t2["unseen_recall"] or 0) >= UNSEEN_MIN,
              f"distractor hits <= {DISTRACTOR_MAX}%": (t2["distractor_hit_rate"] or 0) <= DISTRACTOR_MAX,
              f"v0 unseen-kind recall >= {V0_UNSEEN_MIN}": (t0["unseen_recall"] or 0) >= V0_UNSEEN_MIN,
              f"v0 invented <= {V0_INVENTED_MAX}%": t0["invented_rate"] <= V0_INVENTED_MAX}
    if base:
        gain = t2["fact_recall"] - base["res"]["v2"]["fact_recall"]
        checks[f"recall gain >= +{BAR}"] = gain >= BAR
    print(f"\nholdout verdict for {a.revision}:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not base:
        print("  (recall-gain bar skipped: pass --base)")
    print("HOLDOUT PASS" if all(checks.values()) else "HOLDOUT FAIL")

    out = Path(a.out) / a.revision
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"revision": a.revision, "n": a.n, "tuned": tuned["res"], "base": base["res"] if base else None,
         "checks": checks}, indent=1))
    for k, rows in sets.items():  # kept for the record only; see the rules in the docstring
        with open(out / f"holdout_{k}.jsonl", "w", encoding="utf-8") as fh:
            for r, o in zip(rows, tuned["outs"][k]):
                fh.write(json.dumps({"id": r["id"], "tuned": o["summary"]}, ensure_ascii=False) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
