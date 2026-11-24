# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Generate delbina-sum dataset v0: synthetic agent chats with planted facts.

Each record is the input a delbina summarizer receives (``lines``, the same
"role: text" format as Compressor._summary_lines, optionally starting with
"previous summary: ...") plus the facts a good summary must keep. No model is
involved; summaries are added later by the teacher run.

What makes it harder than bench/make_agent_session.py:
- 13 fact kinds, 2-8 facts per chat, several projects per chat
- facts said by the user, by the assistant, or buried inside tool output
- facts that change later (only the latest value counts; old ones are "stale")
- a previous summary to merge, whose facts may also change
- [[SECRET_n]] placeholders, as delbina's mask mode sends them

Held out for the test split only: 2 fact kinds, the last phrasing of every
template list, and a separate set of project names.

    python datagen/make_dataset.py --out data/v0
    python datagen/make_dataset.py --peek 1
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from delbina import redact_text

VERSION = "v0"
TRAIN_PROJECTS = ["orion", "lyra", "vega", "draco", "hydra", "cygnus", "aquila", "carina", "pavo", "lupus",
                  "atlas", "nova", "sirius", "rigel", "altair", "deneb", "mira", "castor", "pollux", "spica"]
TEST_PROJECTS = ["kestrel", "osprey", "heron", "falcon", "merlin", "harrier", "condor", "plover"]
TEST_KINDS = {"ticket", "env_var"}
PEOPLE = ["Priya", "Arjun", "Meera", "Rahul", "Ananya", "Vikram", "Sneha", "Karthik", "Divya", "Rohan",
          "Lena", "Marco", "Yuki", "Omar", "Sara", "Tomas"]
LIBS = ["requests", "fastapi", "pydantic", "sqlalchemy", "httpx", "numpy", "uvicorn", "boto3", "pandas", "jinja2"]
BACKENDS = ["postgres", "sqlite", "redis", "kafka", "rabbitmq", "minio", "mongodb", "clickhouse"]
COMPONENTS = ["queue", "cache", "storage", "database", "event bus"]
WORDS = "value result config client token session request handler payload retry index buffer item record".split()
TRIM_MARK = "[... delbina trimmed {n} tokens of tool output ...]"


# --- fact kinds ---------------------------------------------------------------------------
# Each kind: label (used in updates and previous summaries), value maker, statement
# templates as (role, text). role "tool" means the line is buried in a tool output.
# The LAST template of every list is held out for the test split.

def _alnum(rng: random.Random, k: int) -> str:
    while True:
        v = "".join(rng.choices(string.ascii_uppercase + string.digits, k=k))
        if any(c.isdigit() for c in v) and any(c.isalpha() for c in v):
            return v


KINDS: dict[str, dict] = {
    "port": {
        "label": "port",
        "value": lambda r, x: str(r.randint(1024, 65000)),
        "say": [("user", "Note for later: the {p} service listens on port {v}."),
                ("assistant", "I checked the compose file: {p} is exposed on port {v}."),
                ("tool", "src/{p}/settings.py:12: PORT = {v}"),
                ("user", "fyi {p} runs on :{v} in staging, don't touch it")],
    },
    "deploy_key": {
        "label": "deploy key",
        "value": lambda r, x: _alnum(r, 6),
        "say": [("user", "The deploy key for project {p} is {v}. Keep it handy."),
                ("assistant", "Got it, deploy key for {p}: {v}."),
                ("user", "{p} deploys need the key {v}, the old one was revoked")],
    },
    "badge_id": {
        "label": "on-call badge ID",
        "value": lambda r, x: f"B-{r.randint(10000, 99999)}",
        "say": [("user", "The on-call engineer for {p} this week has badge ID {v}."),
                ("assistant", "Paging rota says {p} on-call badge is {v}."),
                ("user", "if {p} breaks tonight, escalate to badge {v}")],
    },
    "path": {
        "label": "config path",
        "value": lambda r, x: f"src/{x['p']}/{r.choice(['config', 'settings', 'deploy'])}/"
                              f"{r.choice(WORDS)}.{r.choice(['yaml', 'toml', 'json'])}",
        "say": [("user", "The real config for {p} lives in {v}, ignore the others."),
                ("assistant", "Found it: {p} loads its settings from {v}."),
                ("tool", "src/{p}/main.py:8: CONFIG_PATH = \"{v}\""),
                ("user", "edit {v} for {p}, that's the file prod reads")],
    },
    "version": {
        "label": "{lib} version",
        "extra": lambda r, p: {"lib": r.choice(LIBS)},
        "value": lambda r, x: f"{r.randint(1, 4)}.{r.randint(0, 40)}.{r.randint(0, 12)}",
        "say": [("user", "Pin {lib} to {v} in {p}, newer releases break the build."),
                ("assistant", "{p} works with {lib} {v}; I'll keep that pin."),
                ("tool", "{p}/requirements.txt:4: {lib}=={v}"),
                ("user", "{p} must stay on {lib} {v} until the migration is done")],
    },
    "owner": {
        "label": "owner",
        "value": lambda r, x: r.choice(PEOPLE),
        "say": [("user", "{v} owns the {p} module, ask them before merging."),
                ("assistant", "CODEOWNERS lists {v} as the owner of {p}."),
                ("user", "anything in {p} needs a review from {v}")],
    },
    "decision": {
        "label": "{comp} backend",
        "extra": lambda r, p: {"comp": r.choice(COMPONENTS)},
        "value": lambda r, x: r.choice(BACKENDS),
        "say": [("user", "Decision: {p} will use {v} for its {comp}."),
                ("assistant", "Agreed, we go with {v} as the {comp} for {p}."),
                ("user", "final call on the {p} {comp}: {v}")],
    },
    "deadline": {
        "label": "release freeze date",
        "value": lambda r, x: (date(2026, 12, 1) + timedelta(days=r.randint(0, 120))).isoformat(),
        "say": [("user", "Release freeze for {p} starts on {v}."),
                ("assistant", "Noted: no {p} merges after {v}."),
                ("user", "{p} has to be feature-complete by {v}")],
    },
    "host": {
        "label": "staging host",
        "value": lambda r, x: f"{x['p']}-{r.choice(['stg', 'qa', 'canary'])}-{r.randint(1, 9):02d}.internal",
        "say": [("user", "Staging for {p} is on {v}."),
                ("assistant", "The {p} staging host resolves to {v}."),
                ("tool", "deploy/{p}.env:3: HOST={v}"),
                ("user", "ssh into {v} to check {p} staging")],
    },
    "limit": {
        "label": "rate limit (requests per minute)",
        "value": lambda r, x: str(r.randint(101, 999)),
        "say": [("user", "The {p} API allows {v} requests per minute, stay under it."),
                ("assistant", "{p} is rate-limited to {v} requests/min according to the docs."),
                ("user", "keep {p} calls below {v} per minute or we get banned")],
    },
    "secret": {
        "label": "API key",
        "value": None,  # filled with a [[SECRET_n]] placeholder
        "say": [("user", "The {p} API key is {v}, use it for the integration test."),
                ("assistant", "I'll pass {v} as the {p} API key."),
                ("user", "use {v} to auth against {p}")],
    },
    "ticket": {  # test only
        "label": "tracking ticket",
        "value": lambda r, x: f"{r.choice(['OPS', 'BUG', 'INC'])}-{r.randint(100, 9999)}",
        "say": [("user", "The {p} outage is tracked in {v}."),
                ("assistant", "Linked the {p} fix to {v}."),
                ("user", "put {v} in the commit message for {p}")],
    },
    "env_var": {  # test only
        "label": "feature flag",
        "value": lambda r, x: f"{x['p'].upper()}_{r.choice(WORDS).upper()}_ENABLED",
        "say": [("user", "Turn on {v} to enable the new {p} path."),
                ("tool", "deploy/{p}.env:7: {v}=1"),
                ("user", "{p} reads the flag {v}, set it in staging only")],
    },
}
CHANGEABLE = ["port", "deploy_key", "path", "version", "owner", "decision", "deadline", "host", "limit"]
UPDATES = [
    ("user", "Correction: the {label} for {p} is now {v} (it was {old})."),
    ("user", "Update: {p} {label} changed to {v}."),
    ("assistant", "Heads up, the {p} {label} moved from {old} to {v}."),
    ("user", "scratch that, {p} {label} should be {v}"),
]


# --- filler -------------------------------------------------------------------------------
CHAT = [
    ("Why does the {p} test hang on CI?", "It waits on a socket that CI never opens; I added a timeout."),
    ("Can you explain what the retry decorator in {p} does?", "It retries the call up to three times with backoff and re-raises the last error."),
    ("Is the {p} cache safe to clear?", "Yes, it rebuilds on the next request; only the first call gets slower."),
    ("Rename the helper in {p} to something clearer.", "Renamed it and updated the two call sites."),
    ("What's left on the {p} refactor?", "Only the error messages; the logic is done."),
]
ACKS = [("user", "ok thanks"), ("assistant", "Done."), ("user", "looks good, continue"), ("assistant", "On it.")]


def _tool_output(rng: random.Random, p: str, buried: str | None) -> str:
    rows = [f"src/{p}/{rng.choice(['core', 'util', 'api'])}.py:{rng.randint(1, 99)}: "
            f"{rng.choice(['def', 'return', 'raise', 'log'])} {rng.choice(WORDS)}_{rng.choice(WORDS)}"
            for _ in range(rng.randint(4, 18))]
    if buried:
        rows.insert(rng.randint(0, len(rows)), buried)
    if rng.random() < 0.3:
        rows.insert(len(rows) // 2, TRIM_MARK.format(n=rng.randint(200, 3000)))
    return "\n".join(rows)


def _tool_step(rng: random.Random, p: str, buried: str | None = None) -> list[str]:
    tool, args = rng.choice([("grep", {"pattern": f"{p}_{rng.choice(WORDS)}"}),
                             ("read_file", {"path": f"src/{p}/{rng.choice(['core', 'util', 'api'])}.py"}),
                             ("run_tests", {"path": f"tests/test_{p}.py"})])
    lines = [f"assistant: {tool}({json.dumps(args)})", f"tool: {_tool_output(rng, p, buried)}"]
    if rng.random() < 0.6:
        lines.append(f"assistant: Checked {p}, nothing unusual there.")
    return lines


def _filler(rng: random.Random, projects: list[str]) -> list[str]:
    p = rng.choice(projects)
    roll = rng.random()
    if roll < 0.55:
        return [f"user: Look into the {p} module."] + _tool_step(rng, p)
    if roll < 0.85:
        q, a = rng.choice(CHAT)
        return [f"user: {q.format(p=p)}", f"assistant: {a}"]
    role, text = rng.choice(ACKS)
    return [f"{role}: {text}"]


# --- one sample ---------------------------------------------------------------------------

def make_sample(split: str, i: int) -> dict:
    rng = random.Random(f"delbina-sum-{VERSION}-{split}-{i}")
    test = split == "test"
    pool = TEST_PROJECTS if test else TRAIN_PROJECTS
    kinds = list(KINDS) if test else [k for k in KINDS if k not in TEST_KINDS]
    projects = rng.sample(pool, rng.randint(2, 4))

    def pick(templates: list) -> tuple[str, str]:
        usable = templates if test else templates[:-1]
        return rng.choice(usable)

    # choose facts: unique (kind, project) pairs, unique values
    n_facts = rng.randint(2, 8)
    facts, used, secrets = [], set(), 0
    while len(facts) < n_facts:
        kind, p = rng.choice(kinds), rng.choice(projects)
        if (kind, p) in used:
            continue
        spec = KINDS[kind]
        extra = {"p": p, **(spec["extra"](rng, p) if "extra" in spec else {})}
        if kind == "secret":
            secrets += 1
            value = f"[[SECRET_{secrets}]]"
        else:
            value = spec["value"](rng, extra)
        if any(value == f["value"] for f in facts):
            continue
        used.add((kind, p))
        facts.append({"kind": kind, "subject": p, "value": value, "stale": [], "extra": extra})

    # changes: some facts get a newer value later
    for f in facts:
        if f["kind"] in CHANGEABLE and rng.random() < 0.3:
            spec = KINDS[f["kind"]]
            for _ in range(10):
                new = spec["value"](rng, f["extra"])
                taken = {v for g in facts for v in [g["value"], *g["stale"]]}
                if new not in taken:
                    f["stale"].append(f["value"])
                    f["value"] = new
                    break

    # previous summary: some facts arrive already summarized
    prev = [f for f in facts if rng.random() < 0.25] if rng.random() < 0.35 else []
    prev_parts = []
    for f in prev:
        label = KINDS[f["kind"]]["label"].format(**f["extra"])
        first = f["stale"][0] if f["stale"] else f["value"]
        prev_parts.append(f"{f['subject']} {label}: {first}")
        f["where"] = "previous"

    # build the chat: filler steps with fact statements and updates spread across it
    steps = rng.randint(4, 28)
    events: dict[int, list[tuple[str, dict]]] = {}
    for f in facts:
        at = -1 if f in prev else rng.randrange(steps)
        if at >= 0:
            events.setdefault(at, []).append(("say", f))
            f["where"] = ("early", "middle", "late")[min(at * 3 // steps, 2)]
        if f["stale"]:
            later = rng.randint(max(at + 1, 0), steps)
            events.setdefault(later, []).append(("update", f))

    lines = [f"previous summary: The work so far covered {', '.join(projects)}. Known facts: "
             + "; ".join(prev_parts) + "."] if prev_parts else []
    for s in range(steps + 1):
        for action, f in events.get(s, []):
            ctx = {**f["extra"], "v": f["stale"][0] if action == "say" and f["stale"] else f["value"]}
            if action == "say":
                role, text = pick(KINDS[f["kind"]]["say"])
                text = text.format(**ctx)
                lines += (_tool_step(rng, f["subject"], buried=text) if role == "tool" else [f"{role}: {text}"])
            else:
                role, text = pick(UPDATES)
                label = KINDS[f["kind"]]["label"].format(**f["extra"])
                lines.append(f"{role}: " + text.format(label=label, old=f["stale"][0], **ctx))
            if rng.random() < 0.5:
                lines.append(rng.choice(["assistant: Noted.", "assistant: Got it.", "assistant: OK."]))
        if s < steps:
            lines += _filler(rng, projects)

    out_facts = [{"kind": f["kind"], "subject": f["subject"], "value": f["value"], "stale": f["stale"],
                  "where": f["where"]} for f in facts]
    sample = {
        "id": f"{VERSION}-{split}-{i:06d}",
        "split": split,
        "lines": lines,
        "facts": out_facts,
        "n_chars": len("\n".join(lines)),  # ~4 chars/token; Qwen token counts come later
        "has_previous": bool(prev_parts),
        "source": "synthetic",
        "generator": f"delbina datagen {VERSION}",
        "license": "Apache-2.0",
    }
    verify(sample)
    return sample


def verify(sample: dict) -> None:
    """Every planted value is in the input, and redaction would not alter the input."""
    text = "\n".join(sample["lines"])
    for f in sample["facts"]:
        for v in [f["value"], *f["stale"]]:
            assert v in text, f"{sample['id']}: {v!r} missing from lines"
    for line in sample["lines"]:
        assert redact_text(line) == line, f"{sample['id']}: redaction changed {line[:80]!r}"


# --- CLI ----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/" + VERSION)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--peek", type=int, metavar="N", help="print N train samples and exit")
    a = ap.parse_args(argv)

    if a.peek:
        for i in range(a.peek):
            print(json.dumps(make_sample("train", i), indent=1, ensure_ascii=False))
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"version": VERSION}
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        kinds, sizes = Counter(), []
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(n):
                s = make_sample(split, i)
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                kinds.update(f["kind"] for f in s["facts"])
                sizes.append(s["n_chars"])
        sizes.sort()
        stats[split] = {"samples": n, "facts": sum(kinds.values()), "kinds": dict(sorted(kinds.items())),
                        "chars_p50": sizes[n // 2] if n else 0, "chars_p95": sizes[int(n * 0.95)] if n else 0,
                        "chars_max": sizes[-1] if n else 0}
        print(f"{split:5} {n:5} samples  {stats[split]['facts']:6} facts  "
              f"chars p50 {stats[split]['chars_p50']}  p95 {stats[split]['chars_p95']}", file=sys.stderr)
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
