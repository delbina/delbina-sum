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


def test_target_correct_by_construction():
    for s in TRAIN + TEST:
        assert score(s["target"], s["facts"])["ok"], s["id"]
        fact_lines = [l for l in s["target"].splitlines() if l.startswith("- ") and l.count(": ") >= 2]
        for f in s["facts"]:  # each value sits on a line naming its own project
            assert f"- {f['subject']}: {f['label']}: {f['value']}" in fact_lines, s["id"]
        ctx = s["target"].split("Context:\n")[1:] or [""]
        assert ctx[0].count("\n- ") < md.MAX_NOTES, s["id"]


def test_score_binding():
    facts = [{"subject": "atlas", "value": "src/atlas/settings/config.toml", "stale": []},
             {"subject": "pavo", "value": "pavo-qa-09.internal", "stale": []}]
    good = "- atlas: config path: src/atlas/settings/config.toml\n- pavo: staging host: pavo-qa-09.internal"
    assert score(good, facts)["ok"]
    # wrong project: value contains "atlas" but the line names hydra
    r = score("- hydra: config path: src/atlas/settings/config.toml\n- pavo: host: pavo-qa-09.internal", facts)
    assert r["recall"] == 1 and r["bound_recall"] == 0.5 and r["unbound"] == ["src/atlas/settings/config.toml"]
    # project-less mention
    r = score("staging host pavo-qa-09.internal\natlas uses src/atlas/settings/config.toml", facts)
    assert r["unbound"] == ["pavo-qa-09.internal"] and not r["ok"]
