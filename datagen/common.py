# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Shared by the generator, the teacher run and the eval: prompts and fact scoring.

The student is trained on the exact production prompt (delbina.ollama.PROMPT), so the
fine-tuned model drops straight into ``openai_summarizer``. The teacher gets the same
prompt plus stricter rules, which only shape the targets.
"""
from __future__ import annotations

import re

from delbina.ollama import PROMPT

TEACHER_RULES = (
    "Rules: copy every identifier, number, version, date, path, host, ID and placeholder "
    "like [[SECRET_1]] exactly as written. When a value was changed later, keep only the "
    "latest value and do not mention the old one. Say which project or module each fact "
    "belongs to. No preamble."
)


def student_prompt(lines: list[str]) -> str:
    """Exactly what delbina sends a summarizer in production."""
    return PROMPT.format(chat="\n".join(f"- {t}" for t in lines))


def teacher_prompt(lines: list[str]) -> str:
    return student_prompt(lines).replace("\n\nSummary:", f"\n\n{TEACHER_RULES}\n\nSummary:")


def _has(text: str, value: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text) is not None


def score(summary: str, facts: list[dict]) -> dict:
    """Fact recall of a summary. A fact counts if its current value appears verbatim.
    ``stale_hits`` counts superseded values that still appear (should be 0)."""
    missing = [f["value"] for f in facts if not _has(summary, f["value"])]
    stale = [s for f in facts for s in f.get("stale", ()) if _has(summary, s)]
    n = len(facts)
    return {
        "recall": (n - len(missing)) / n if n else 1.0,
        "missing": missing,
        "stale_hits": stale,
        "ok": not missing and not stale,
    }
