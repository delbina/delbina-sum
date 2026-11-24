# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""LoRA fine-tuning and evaluation helpers for delbina-sum (used by notebooks/train.ipynb).

Training pairs: user turn = the exact production prompt (common.student_prompt), assistant
turn = the sample's correct-by-construction ``target``. Loss is on the assistant turn only.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from common import clean_summary, score, student_prompt


def _prompt_text(tok, lines: list[str]) -> str:
    return tok.apply_chat_template([{"role": "user", "content": student_prompt(lines)}],
                                   tokenize=False, add_generation_prompt=True)


def build_examples(tok, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    """Tokenize rows into input_ids/labels with the prompt masked out. Returns (examples, n_dropped)."""
    out, dropped = [], 0
    for r in rows:
        prompt = _prompt_text(tok, r["lines"])
        full = tok.apply_chat_template([{"role": "user", "content": student_prompt(r["lines"])},
                                        {"role": "assistant", "content": r["target"]}], tokenize=False)
        assert full.startswith(prompt), "chat template is not prefix-stable"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append({"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids})
    return out, dropped


class Collator:
    """Right-pad a batch; padded label positions are ignored by the loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


@torch.no_grad()
def summarize(model, tok, rows: list[dict], batch_size: int = 8, max_new_tokens: int = 384) -> list[dict]:
    """Greedy summaries for rows, batched by prompt length. Returns [{"summary", "truncated"}] in row order."""
    model.eval()
    tok.padding_side = "left"
    prompts = [_prompt_text(tok, r["lines"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    results: list[dict | None] = [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, row in zip(idx, new):
            n_tok = int((row != tok.pad_token_id).sum())
            text = tok.decode(row, skip_special_tokens=True)
            results[i] = {"summary": clean_summary(text), "truncated": n_tok >= max_new_tokens}
    return results  # type: ignore[return-value]


def evaluate(rows: list[dict], outputs: list[dict]) -> dict:
    """Aggregate fact-level metrics over a split (same definitions as the teacher notebook)."""
    by_kind, by_where = defaultdict(list), defaultdict(list)
    hits, bound, stale, ok, trunc, chars = [], [], [], [], [], []
    for r, o in zip(rows, outputs):
        sc = score(o["summary"], r["facts"])
        missing, unbound = set(sc["missing"]), set(sc["unbound"])
        for f in r["facts"]:
            h = f["value"] not in missing
            hits.append(h); bound.append(h and f["value"] not in unbound)
            by_kind[f["kind"]].append(h); by_where[f["where"]].append(h)
        stale.append(bool(sc["stale_hits"])); ok.append(sc["ok"])
        trunc.append(o["truncated"]); chars.append(len(o["summary"]))
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    chars.sort()
    return {
        "samples": len(rows),
        "fact_recall": pct(hits),
        "bound_recall": pct(bound),
        "stale_hit_rate": pct(stale),
        "perfect_rate": pct(ok),
        "truncated": sum(trunc),
        "recall_by_kind": {k: pct(v) for k, v in sorted(by_kind.items())},
        "recall_by_where": {k: pct(v) for k, v in sorted(by_where.items())},
        "summary_chars_p50": chars[len(chars) // 2] if chars else 0,
    }
