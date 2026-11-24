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

TEACHER_RULES = """Rules for the summary:
1. Start with "Facts:" and one line per specific value: keys, IDs, ports, numbers, limits, versions, dates, paths, hosts, people, chosen tools or backends, and placeholders like [[SECRET_1]]. Write each line as "<project>: <what it is>: <value>", value copied exactly as written.
2. If a value was changed, write only the newest value. Never mention the old value, not even as "previously" or "was".
3. Then "Context:" with at most 3 short lines on decisions or open problems that are not already facts.
4. Leave out tool calls, file listings, grep and test output, and routine steps like "checked X, nothing unusual".
5. No other headings, no preamble."""


def student_prompt(lines: list[str]) -> str:
    """Exactly what delbina sends a summarizer in production."""
    return PROMPT.format(chat="\n".join(f"- {t}" for t in lines))


def teacher_prompt(lines: list[str]) -> str:
    return student_prompt(lines).replace("\n\nSummary:", f"\n\n{TEACHER_RULES}\n\nSummary:")


def clean_summary(text: str) -> str:
    """Strip a leading "Summary:" that models like to echo from the prompt."""
    return re.sub(r"^\s*summary\s*:\s*", "", text.strip(), flags=re.IGNORECASE)


def _has(text: str, value: str) -> bool:
    """Value present as a whole token, ignoring case ("MongoDB" counts for "mongodb")."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


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
