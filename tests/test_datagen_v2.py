# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))

import make_dataset_v2 as md  # noqa: E402
from common import _has, fact_lines, invented, score  # noqa: E402

TRAIN = [md.make_sample("train", i) for i in range(400)]
TEST = [md.make_sample("test", i) for i in range(300)]


def test_deterministic():
    assert md.make_sample("train", 7) == md.make_sample("train", 7)


def test_held_out_kinds_projects_labels():
    train_kinds = {f["kind"] for s in TRAIN for f in s["facts"] + s["open"]}
    assert not train_kinds & md.TEST_KINDS
    assert len(train_kinds) == len(md.KINDS) - len(md.TEST_KINDS)
    assert {f["kind"] for s in TEST for f in s["facts"]} >= md.TEST_KINDS
    text = "\n".join("\n".join(s["lines"]) for s in TRAIN)
    assert not any(p in text for p in md.TEST_PROJECTS)
    held = {(k, spec["labels"][-1]) for k, spec in md.KINDS.items() if "{" not in spec["labels"][-1]}
    assert not {(f["kind"], f["label"]) for s in TRAIN for f in s["facts"]} & held


def test_held_out_templates():
    lists = [md.GENERIC, md.UPDATES, md.ASK] + [s["say"] for s in md.KINDS.values()]
    for tpls in lists:
        stem = tpls[-1][1].split("{")[0]
        if len(stem) >= 8:
            assert not any(stem in line for s in TRAIN for line in s["lines"]), stem


def test_values_unique_and_not_nested():
    for s in TRAIN + TEST:
        vals = [v for f in s["facts"] for v in [f["value"], *f["stale"]]] + s["distractors"]
        assert len(vals) == len(set(vals)), s["id"]
        for a in vals:
            for b in vals:
                assert a == b or not _has(b, a), (s["id"], a, b)


def test_target_shape():
    for s in TRAIN + TEST:
        assert score(s["target"], s["facts"], s["distractors"])["ok"], s["id"]
        lines = fact_lines(s["target"])
        assert len(lines) == len(s["facts"]), s["id"]
        assert not invented(s["target"], s["lines"]), s["id"]
        for f in s["facts"]:
            assert (f["subject"], f["label"], f["value"]) in lines, s["id"]
        ctx = s["target"].split("Context:\n")[1].splitlines() if "Context:\n" in s["target"] else []
        assert len(ctx) <= md.MAX_NOTES, s["id"]
        for o in s["open"]:  # every open item is noted, none becomes a fact line
            assert f"- {md.open_note(o)}" in ctx, s["id"]


def test_mix():
    n = len(TRAIN)
    assert sum(bool(s["open"]) for s in TRAIN) > n * 0.4
    assert sum(bool(s["distractors"]) for s in TRAIN) > n * 0.2
    assert sum(s["lines"][0].startswith("previous summary: Facts:") for s in TRAIN) > n * 0.1
    assert sum(any(f["stale"] for f in s["facts"]) for s in TRAIN) > n * 0.4


def test_score_distractors():
    facts = [{"value": "8080", "stale": []}]
    assert score("p: port: 8080", facts, ["9090"])["ok"]
    r = score("p: port: 8080, maybe 9090", facts, ["9090"])
    assert r["distractor_hits"] == ["9090"] and not r["ok"]
    assert score("p: port: 8080", facts)["distractor_hits"] == []  # v0 rows have no distractors


def test_fact_lines_and_invented():
    lines = ["user: orion port is 8080", "user: key K7Q2ZP for lyra"]
    summ = ("Facts:\n- orion: port: 8080\n- lyra: deploy key: `K7Q2ZP`.\n- vega: ticket: OPS-1\n"
            "Context:\n- orion: a: b: c")
    assert fact_lines(summ) == [("orion", "port", "8080"), ("lyra", "deploy key", "K7Q2ZP"), ("vega", "ticket", "OPS-1")]
    assert invented(summ, lines) == ["OPS-1"]
    assert invented("- orion: port: 8080 (was 7070)", lines) == ["8080 (was 7070)"]
    assert invented("no fact lines here", lines) == []


def test_test_only_styles_and_counts():
    test_stems = [t.split("{")[0] for _, t in md.GENERIC_TEST_ONLY if len(t.split("{")[0]) >= 3]
    assert test_stems
    for stem in test_stems:
        assert not any(line.startswith(f"user: {stem}") or f": {stem}" in line
                       for s in TRAIN for line in s["lines"]), stem
    assert any(f["how"] == "test_style" for s in TEST for f in s["facts"])
    assert not any(f["how"] == "test_style" for s in TRAIN for f in s["facts"])
    assert sum(len(s["distractors"]) for s in TRAIN) > len(TRAIN) * 0.5


def test_v0_test_stays_unseen():
    """The frozen v0 test must stay out-of-generator: none of v0's held-out phrasings in v2 train."""
    import re
    import make_dataset as v0
    held = [spec["say"][-1][1] for spec in v0.KINDS.values()] + [v0.UPDATES[-1][1], v0.CHAT[-1][0]]
    held += [t for k in v0.TEST_KINDS for _, t in v0.KINDS[k]["say"]]
    pats = [re.compile(re.sub(r"\\\{\w+\\\}", ".+?", re.escape(t))) for t in held]
    lines = [line for s in TRAIN for line in s["lines"]]
    for t, pat in zip(held, pats):
        assert not any(pat.search(line) for line in lines), t
    text = "\n".join(lines)
    assert not any(p in text for p in v0.TEST_PROJECTS)
