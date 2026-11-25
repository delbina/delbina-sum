# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Report-only invented metrics: whole-token match and bootstrap CI."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]
sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))

from common import invented, invented_strict, rate_ci  # noqa: E402
import sft  # noqa: E402

CHAT = ["kestrel: the webmaster said port 8081 is untrue for staging"]


def test_strict_catches_values_hidden_inside_longer_tokens():
    summary = "Facts:\n- kestrel: branch: master\n- kestrel: flag: true\n- kestrel: replicas: 1\n- kestrel: port: 8081"
    assert invented(summary, CHAT) == []                        # substring: all "copied"
    assert invented_strict(summary, CHAT) == ["master", "true", "1"]


def test_strict_is_superset():
    summary = "Facts:\n- kestrel: port: 9090\n- kestrel: branch: master"
    assert set(invented(summary, CHAT)) <= set(invented_strict(summary, CHAT))


def test_rate_ci_brackets_rate_and_is_deterministic():
    per = [(1, 10)] * 20 + [(0, 10)] * 80
    lo, hi = rate_ci(per)
    assert lo <= 2.0 <= hi and rate_ci(per) == (lo, hi)
    assert rate_ci([(0, 5)] * 50) == (0.0, 0.0)


def test_evaluate_reports_strict_and_ci():
    rows = [{"lines": CHAT, "facts": [{"subject": "kestrel", "value": "8081", "kind": "port", "where": "early"}]}]
    outs = [{"summary": "Facts:\n- kestrel: port: 8081\n- kestrel: branch: master", "truncated": False}]
    res = sft.evaluate(rows, outs)
    assert res["invented_rate"] == 0.0 and res["invented_strict_rate"] == 50.0
    assert res["fact_lines"] == 2 and len(res["invented_strict_ci"]) == 2
