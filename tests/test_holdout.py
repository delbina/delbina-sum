# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""The holdout split: test distribution, different chats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))

import pytest  # noqa: E402

import make_dataset as md0  # noqa: E402
import make_dataset_v2 as md2  # noqa: E402
import make_dataset_v4 as md4  # noqa: E402


@pytest.mark.parametrize("m", [md0, md2])
def test_holdout_is_test_distribution_not_test_chats(m):
    test = [m.make_sample("test", i) for i in range(100)]
    hold = [m.make_sample("holdout", i) for i in range(100)]
    assert not {tuple(s["lines"]) for s in test} & {tuple(s["lines"]) for s in hold}
    kinds = {f["kind"] for s in hold for f in s["facts"]}
    assert kinds & m.TEST_KINDS, "holdout must include held-out kinds, like test"
    assert all(s["split"] == "holdout" for s in hold)


def test_v4_holdout_is_v2s():
    assert md4.make_sample("holdout", 7) == md2.make_sample("holdout", 7)
