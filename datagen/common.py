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


def _bound(summary: str, subject: str, value: str) -> bool:
    """Value sits on a line that also names its project. The value itself is blanked first,
    because hosts, paths and flags often contain a project name ("pavo-qa-09.internal")."""
    for line in summary.splitlines():
        if _has(line, value):
            rest = re.sub(re.escape(value), " ", line, flags=re.IGNORECASE)
            if _has(rest, subject):
                return True
    return False


def score(summary: str, facts: list[dict], distractors: list[str] = ()) -> dict:
    """Fact recall of a summary. A fact counts if its current value appears verbatim.
    ``unbound``: values present but never on a line naming their own project (wrong-project
    or project-less mentions). ``stale_hits``: superseded values that still appear.
    ``distractor_hits``: rejected proposals that appear (v1). ``ok`` needs all clean."""
    missing = [f["value"] for f in facts if not _has(summary, f["value"])]
    unbound = [f["value"] for f in facts
               if f["value"] not in missing and "subject" in f and not _bound(summary, f["subject"], f["value"])]
    stale = [s for f in facts for s in f.get("stale", ()) if _has(summary, s)]
    dis = [d for d in distractors if _has(summary, d)]
    n = len(facts)
    return {
        "recall": (n - len(missing)) / n if n else 1.0,
        "bound_recall": (n - len(missing) - len(unbound)) / n if n else 1.0,
        "missing": missing,
        "unbound": unbound,
        "stale_hits": stale,
        "distractor_hits": dis,
        "ok": not missing and not unbound and not stale and not dis,
    }


def fact_lines(summary: str) -> list[tuple[str, str, str]]:
    """(subject, label, value) for every "- subject: label: value" line in the Facts part
    (everything before a "Context:" line). Trailing punctuation and backticks are dropped."""
    out = []
    for line in summary.split("\nContext:")[0].splitlines():
        line = line.strip()
        if not line.startswith(("- ", "* ")):
            continue
        parts = line[2:].split(": ", 2)
        if len(parts) == 3:
            value = parts[2].strip().rstrip(".;,").strip("`").strip()
            if value:
                out.append((parts[0].strip(), parts[1].strip(), value))
    return out


def invented(summary: str, lines: list[str]) -> list[str]:
    """Fact-line values that never occur in the input (case-insensitive substring), i.e.
    made up rather than copied. A value with extra words glued on ("8080 (was 7070)")
    also counts: the line no longer carries a clean copied value."""
    text = "\n".join(lines).lower()
    return [v for _, _, v in fact_lines(summary) if v.lower() not in text]


def invented_strict(summary: str, lines: list[str]) -> list[str]:
    """Like ``invented`` but the value must occur as a whole token (``_has``), so "true" inside
    "untrue", "master" inside "webmaster" or "1" inside "8081" no longer count as copied.
    Always a superset of ``invented``. Report-only: the bars still use ``invented``."""
    text = "\n".join(lines)
    return [v for _, _, v in fact_lines(summary) if not _has(text, v)]


def rate_ci(per_chat: list[tuple[int, int]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI, in %, for sum(bad) / sum(lines) over (bad, lines) per chat.
    Resamples whole chats, because the lines of one chat are not independent."""
    import random
    rng, n, stats = random.Random(seed), len(per_chat), []
    for _ in range(n_boot):
        b = l = 0
        for _ in range(n):
            x, y = per_chat[rng.randrange(n)]
            b += x; l += y
        stats.append(100 * b / l if l else 0.0)
    stats.sort()
    return round(stats[int(0.025 * n_boot)], 1), round(stats[int(0.975 * n_boot) - 1], 1)
