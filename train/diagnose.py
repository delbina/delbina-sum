# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Why did a run fail its bars? Breaks down a saved eval (from the HF model repo branch):

  - rejected values (distractors) that leaked, by reject template (seen vs held-out) and by
    where they landed (Facts or Context)
  - values recalled but bound to the wrong project
  - v0-test "invented" values: glued onto a real value ("8080 (was 7070)") or not in the input at all

    python train/diagnose.py                       # downloads eval files from delbina/delbina-sum@v2
    python train/diagnose.py --revision v3
    python train/diagnose.py --v2 eval_test.jsonl --v0 eval_test_v0.jsonl   # local copies
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]

import make_dataset as md0  # noqa: E402
import make_dataset_v2 as md  # noqa: E402
from common import _bound, _has, fact_lines, invented, invented_strict, rate_ci  # noqa: E402

REPO = "delbina/delbina-sum"


def load(path: str | None, name: str, revision: str) -> list[dict]:
    if not path:
        from huggingface_hub import hf_hub_download  # token: HF_TOKEN env or a saved login
        path = hf_hub_download(REPO, name, revision=revision)
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _tpl_regex(role: str, text: str) -> re.Pattern:
    pat = re.escape(f"{role}: {text}")
    for ph in ("p", "label", "d", "v"):
        pat = pat.replace(re.escape("{" + ph + "}"), ".+?")
    return re.compile(f"^{pat}$", re.IGNORECASE | re.DOTALL)


TEMPLATES = [(name, i, i == len(lst) - 1, [_tpl_regex(r, t) for r, t in tpl])
             for name, lst in (("keep", md.REJECT_KEEP), ("open", md.REJECT_OPEN)) for i, tpl in enumerate(lst)]


def which_template(lines: list[str], d: str) -> str:
    for line in (l for l in lines if _has(l, d)):
        for name, i, held, pats in TEMPLATES:
            if any(p.match(line) for p in pats):
                return f"{name}[{i}]{' HELD-OUT' if held else ''}"
    return "unmatched"


def section(summary: str, d: str) -> str:
    facts, _, ctx = summary.partition("\nContext:")
    return "+".join(s for s, t in (("Facts", facts), ("Context", ctx)) if _has(t, d)) or "?"


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def v2_report(recs: list[dict]) -> None:
    rows = [md.make_sample("test", int(r["id"].rsplit("-", 1)[1])) for r in recs]
    assert all(r["id"] == s["id"] for r, s in zip(recs, rows)), "ids don't match the regenerated split"
    seen, hit, where, examples = Counter(), Counter(), Counter(), defaultdict(list)
    wrong_bind = []
    for r, s in zip(recs, rows):
        out = r["tuned"]
        for d in s["distractors"]:
            t = which_template(s["lines"], d)
            seen[t] += 1
            if _has(out, d):
                hit[t] += 1
                where[section(out, d)] += 1
                if len(examples[t]) < 2:
                    line = next(l for l in out.splitlines() if _has(l, d))
                    examples[t].append(f"{s['id']}: {d!r} -> {line.strip()}")
        for f in s["facts"]:
            if _has(out, f["value"]) and not _bound(out, f["subject"], f["value"]):
                line = next(l for l in out.splitlines() if _has(l, f["value"]))
                wrong_bind.append((f["kind"], f"{s['id']}: {f['subject']} {f['value']!r} -> {line.strip()}"))

    chats = sum(1 for s in rows if s["distractors"])
    leaked = sum(1 for r, s in zip(recs, rows) if any(_has(r["tuned"], d) for d in s["distractors"]))
    print(f"== v2 test: rejected values ==\nchats with a reject that leaked: {pct(leaked, chats)}")
    for t in sorted(seen):
        print(f"  {t:22} leaked {pct(hit[t], seen[t])}")
    print("  landed in:", dict(where))
    for t, ex in sorted(examples.items()):
        for e in ex:
            print(f"    [{t}] {e}")
    print(f"\n== v2 test: recalled but bound to the wrong project: {len(wrong_bind)} ==")
    print("  by kind:", dict(Counter(k for k, _ in wrong_bind).most_common(8)))
    for _, e in wrong_bind[:6]:
        print("   ", e)


def v0_report(recs: list[dict], show_all: bool = False) -> None:
    rows = [md0.make_sample("test", int(r["id"].rsplit("-", 1)[1])) for r in recs]
    assert all(r["id"] == s["id"] for r, s in zip(recs, rows)), "ids don't match the v0 test split"
    glued, made_up, token_only = [], [], []
    per, per_strict = [], []
    for r, s in zip(recs, rows):
        inv = invented(r["tuned"], s["lines"])
        strict = invented_strict(r["tuned"], s["lines"])
        n = len(fact_lines(r["tuned"]))
        per.append((len(inv), n)); per_strict.append((len(strict), n))
        token_only += [f"{s['id']}: {v!r}" for v in strict if v not in inv]
        for v in inv:
            real = [f["value"] for f in s["facts"] if f["value"].lower() in v.lower()]
            (glued if real else made_up).append(f"{s['id']}: {v!r}" + (f"  (real: {real[0]!r})" if real else ""))
    n_lines = sum(n for _, n in per)
    rate = lambda p: round(100 * sum(x for x, _ in p) / n_lines, 1) if n_lines else 0.0
    print(f"\n== v0 test: {n_lines} fact lines ==")
    print(f"  invented (substring, the bar): {rate(per)}%  95% CI {list(rate_ci(per))}")
    print(f"  invented (whole token, report): {rate(per_strict)}%  95% CI {list(rate_ci(per_strict))}")
    print(f"  only caught by whole-token match (inside a longer word or number): {len(token_only)}")
    for e in (token_only if show_all else token_only[:8]):
        print("   ", e)
    print(f"\n== v0 test: invented values: {len(glued) + len(made_up)} ==")
    print(f"  glued onto a real value (format problem): {len(glued)}")
    for e in (glued if show_all else glued[:5]):
        print("   ", e)
    print(f"  not in the input at all (made up / mangled): {len(made_up)}")
    for e in (made_up if show_all else made_up[:8]):
        print("   ", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="v2")
    ap.add_argument("--v2", help="local eval_test.jsonl")
    ap.add_argument("--v0", help="local eval_test_v0.jsonl")
    ap.add_argument("--all", action="store_true", help="list every v0 invented value")
    a = ap.parse_args()
    v2_report(load(a.v2, "eval_test.jsonl", a.revision))
    v0_report(load(a.v0, "eval_test_v0.jsonl", a.revision), a.all)


if __name__ == "__main__":
    main()
