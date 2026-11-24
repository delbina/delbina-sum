# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))

import make_dataset as md  # noqa: E402
from common import clean_summary, score, student_prompt, teacher_prompt  # noqa: E402

TRAIN = [md.make_sample("train", i) for i in range(300)]
TEST = [md.make_sample("test", i) for i in range(300)]


def test_deterministic():
    assert md.make_sample("train", 5) == md.make_sample("train", 5)


def test_held_out_kinds_and_projects():
    train_kinds = {f["kind"] for s in TRAIN for f in s["facts"]}
    assert not train_kinds & md.TEST_KINDS
    assert {f["kind"] for s in TEST for f in s["facts"]} & md.TEST_KINDS
    text = "\n".join("\n".join(s["lines"]) for s in TRAIN)
    assert not any(p in text for p in md.TEST_PROJECTS)


def test_held_out_phrasing():
    for kind, spec in md.KINDS.items():
        held = spec["say"][-1][1].split("{")[0]
        if len(held) < 6:
            continue
        assert not any(held in line for s in TRAIN for line in s["lines"]), kind


def test_values_unique_and_present():
    for s in TRAIN + TEST:
        vals = [v for f in s["facts"] for v in [f["value"], *f["stale"]]]
        assert len(vals) == len(set(vals)), s["id"]
        md.verify(s)


def test_score():
    facts = [{"value": "8080", "stale": ["7070"]}, {"value": "K7Q2ZP", "stale": []}]
    assert score("orion port 8080, key K7Q2ZP", facts)["ok"]
    r = score("orion port 18080 (was 7070)", facts)
    assert r["recall"] == 0 and r["stale_hits"] == ["7070"]
    assert score("Decision: use MongoDB", [{"value": "mongodb", "stale": []}])["ok"]
    assert clean_summary("Summary:\n- a") == "- a" and clean_summary("- a") == "- a"


def test_prompts():
    lines = ["user: hi"]
    assert student_prompt(lines).endswith("Summary:")
    assert "newest value" in teacher_prompt(lines) and teacher_prompt(lines).endswith("Summary:")
