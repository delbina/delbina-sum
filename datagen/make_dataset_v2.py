# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Generate delbina-sum dataset v2: v1 plus what the v1 model still missed.

v1 model (93.7 recall on v1 test) fell to 43 unseen-kind recall on the frozen v0 test:
values shaped like nothing it trained on (UPPER_SNAKE flags only ever appeared as keys),
and phrasings outside v1's templates. It also kept 10.4% of rejected values. v2 adds:
- 11 train kinds with new value shapes: UPPER_SNAKE and lower_snake names, RFC-style IDs,
  tags, URLs, CIDRs, percentages, dollar amounts, cron lines, durations, weekday times
- more statement styles (JSON, arrows, dotted keys, bullets) and test-only ones (TOML, Q/A,
  parenthetical) to measure style generalization
- updates as `git diff` lines in tool output
- more rejected proposals: outdated wiki values, example values, up to 2 per chat
- facts record how they were stated (``how``) for per-style recall

Everything v1 held out stays held out (and the frozen v0 test stays untouched by training).

---- v1 notes ----
v0's model learned 11 fixed labels and squeezed every new fact into one of them, inventing
values on the way (73/84 invented values sat in chats with unseen kinds). v1 changes:
- 43 fact kinds (32 train, 11 test-only) with 2-4 label wordings each; the target uses the
  wording said in the chat, so the model learns to copy labels instead of mapping to a list
- generic statement templates shared by every kind, plus kind-specific ones
- value-less mentions ("we still need to pick the X for p"): no fact line, an open item in Context
- questions answered later in the chat
- rejected proposals: a value that is said but must NOT appear (``distractors``)
- previous summaries in v0 prose style or in the model's own output format (what delbina
  actually sends back on the next compression)
- filler never contains a planted value, so recall can't come from noise

Held out for the test split: 11 fact kinds, the last label wording of every kind, the last
template of every list, and the v0 test project names.

    python datagen/make_dataset_v2.py --out data/v2
    python datagen/make_dataset_v2.py --peek 1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from delbina import redact_text

from common import _has, score

VERSION = "v2"
TRAIN_PROJECTS = ["orion", "lyra", "vega", "draco", "hydra", "cygnus", "aquila", "carina", "pavo", "lupus",
                  "atlas", "nova", "sirius", "rigel", "altair", "deneb", "mira", "castor", "pollux", "spica",
                  "ember", "quartz", "tundra", "juniper", "saffron", "monsoon", "basalt", "cobalt"]
TEST_PROJECTS = ["kestrel", "osprey", "heron", "falcon", "merlin", "harrier", "condor", "plover"]
PEOPLE = ["Priya", "Arjun", "Meera", "Rahul", "Ananya", "Vikram", "Sneha", "Karthik", "Divya", "Rohan",
          "Lena", "Marco", "Yuki", "Omar", "Sara", "Tomas"]
LIBS = ["requests", "fastapi", "pydantic", "sqlalchemy", "httpx", "numpy", "uvicorn", "boto3", "pandas", "jinja2"]
BACKENDS = ["postgres", "sqlite", "redis", "kafka", "rabbitmq", "minio", "mongodb", "clickhouse"]
COMPONENTS = ["queue", "cache", "storage", "database", "event bus"]
WORDS = "value result config client token session request handler payload retry index buffer item record".split()
TRIM_MARK = "[... delbina trimmed {n} tokens of tool output ...]"
MAX_NOTES = 3
MAX_OPEN = 2


def _alnum(r: random.Random, k: int) -> str:
    while True:
        v = "".join(r.choices(string.ascii_uppercase + string.digits, k=k))
        if any(c.isdigit() for c in v) and any(c.isalpha() for c in v):
            return v


def _day(r: random.Random) -> date:
    return date(2026, 12, 1) + timedelta(days=r.randint(0, 150))


def K(labels, value, say=(), extra=None, changeable=True, secret=False, test=False) -> dict:
    """One fact kind. ``labels``: wordings (last held out for test). ``value(r, x)``: x has "p" and
    the extras. ``say``: kind-specific templates (role, text); last held out. Templates may use
    {p} {v} {label} and the extras."""
    return {"labels": list(labels), "value": value, "say": list(say), "extra": extra,
            "changeable": changeable and not secret, "secret": secret, "test": test}


KINDS: dict[str, dict] = {
    # --- v0 kinds (train) ---
    "port": K(["port", "listen port", "service port"], lambda r, x: str(r.randint(1024, 65000)), [
        ("user", "Note for later: the {p} service listens on port {v}."),
        ("tool", "src/{p}/settings.py:12: PORT = {v}"),
        ("user", "fyi {p} runs on :{v} in staging, don't touch it")]),
    "deploy_key": K(["deploy key", "deployment key"], lambda r, x: _alnum(r, 6), [
        ("user", "The deploy key for project {p} is {v}. Keep it handy."),
        ("user", "{p} deploys need the key {v}, the old one was revoked")]),
    "badge_id": K(["on-call badge ID", "on-call badge", "pager badge"], lambda r, x: f"B-{r.randint(10000, 99999)}", [
        ("assistant", "Paging rota says {p} on-call badge is {v}."),
        ("user", "if {p} breaks tonight, escalate to badge {v}")]),
    "path": K(["config path", "config file", "settings file"],
              lambda r, x: f"src/{x['p']}/{r.choice(['config', 'settings', 'deploy'])}/"
                           f"{r.choice(WORDS)}.{r.choice(['yaml', 'toml', 'json'])}", [
        ("assistant", "Found it: {p} loads its settings from {v}."),
        ("tool", "src/{p}/main.py:8: CONFIG_PATH = \"{v}\""),
        ("user", "edit {v} for {p}, that's the file prod reads")]),
    "version": K(["{lib} version", "{lib} pin", "pinned {lib} version"],
                 lambda r, x: f"{r.randint(1, 4)}.{r.randint(0, 40)}.{r.randint(0, 12)}", [
        ("user", "Pin {lib} to {v} in {p}, newer releases break the build."),
        ("tool", "{p}/requirements.txt:4: {lib}=={v}"),
        ("user", "{p} must stay on {lib} {v} until the migration is done")],
                 extra=lambda r: {"lib": r.choice(LIBS)}),
    "owner": K(["owner", "module owner", "code owner"], lambda r, x: r.choice(PEOPLE), [
        ("user", "{v} owns the {p} module, ask them before merging."),
        ("assistant", "CODEOWNERS lists {v} as the owner of {p}."),
        ("user", "anything in {p} needs a review from {v}")]),
    "decision": K(["{comp} backend", "{comp}", "{comp} choice"], lambda r, x: r.choice(BACKENDS), [
        ("user", "Decision: {p} will use {v} for its {comp}."),
        ("assistant", "Agreed, we go with {v} as the {comp} for {p}."),
        ("user", "final call on the {p} {comp}: {v}")],
                  extra=lambda r: {"comp": r.choice(COMPONENTS)}),
    "deadline": K(["release freeze date", "code freeze", "freeze date"], lambda r, x: _day(r).isoformat(), [
        ("user", "Release freeze for {p} starts on {v}."),
        ("assistant", "Noted: no {p} merges after {v}."),
        ("user", "{p} has to be feature-complete by {v}")]),
    "host": K(["staging host", "staging server", "staging box"],
              lambda r, x: f"{x['p']}-{r.choice(['stg', 'qa', 'canary'])}-{r.randint(1, 9):02d}.internal", [
        ("tool", "deploy/{p}.env:3: HOST={v}"),
        ("user", "ssh into {v} to check {p} staging")]),
    "limit": K(["rate limit (requests per minute)", "rate limit", "requests per minute cap"],
               lambda r, x: str(r.randint(101, 999)), [
        ("user", "The {p} API allows {v} requests per minute, stay under it."),
        ("user", "keep {p} calls below {v} per minute or we get banned")]),
    "secret": K(["API key", "API token", "access key"], None, [
        ("user", "The {p} API key is {v}, use it for the integration test."),
        ("user", "use {v} to auth against {p}")], secret=True),
    # --- new train kinds ---
    "timeout": K(["request timeout", "HTTP timeout", "upstream timeout"], lambda r, x: f"{r.randint(11, 600)}s", [
        ("user", "{p} requests should give up after {v}."),
        ("user", "bump the {p} timeout to {v}, the import is slow")]),
    "memory": K(["memory limit", "container memory", "pod memory limit"],
                lambda r, x: f"{r.choice([256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144])}Mi", [
        ("tool", "k8s/{p}/deployment.yaml:41: memory: {v}"),
        ("user", "{p} keeps getting OOM-killed, give it {v}")]),
    "db_name": K(["database name", "DB name", "schema name"], lambda r, x: f"{x['p']}_{r.choice(WORDS)}_db", [
        ("tool", "src/{p}/db.py:5: DB_NAME = \"{v}\""),
        ("user", "{p} writes everything into {v}")]),
    "bucket": K(["S3 bucket", "storage bucket", "upload bucket"],
                lambda r, x: f"{x['p']}-{r.choice(WORDS)}-{_alnum(r, 4).lower()}", [
        ("user", "{p} uploads go to the bucket {v}."),
        ("tool", "infra/{p}.tf:22: bucket = \"{v}\"")]),
    "branch": K(["release branch", "base branch", "target branch"],
                lambda r, x: f"release/{r.randint(24, 27)}.{r.randint(1, 12)}", [
        ("user", "Open the {p} PRs against {v}."),
        ("user", "{p} hotfixes go to {v}, not main")]),
    "queue": K(["queue name", "topic", "Kafka topic"], lambda r, x: f"{x['p']}.{r.choice(WORDS)}.v{r.randint(1, 9)}", [
        ("assistant", "{p} publishes its events on {v}."),
        ("user", "consume {v} for the {p} sync job")]),
    "image": K(["image tag", "docker image", "container image"],
               lambda r, x: f"ghcr.io/acme/{x['p']}:{r.randint(1, 5)}.{r.randint(0, 20)}.{r.randint(0, 9)}", [
        ("tool", "k8s/{p}/deployment.yaml:19: image: {v}"),
        ("user", "roll {p} back to {v}")]),
    "channel": K(["alerts channel", "Slack channel", "alert channel"],
                 lambda r, x: f"#{x['p']}-{r.choice(['alerts', 'oncall', 'deploys', 'incidents'])}", [
        ("user", "{p} alerts go to {v}."),
        ("user", "post the {p} rollout status in {v}")]),
    "account_id": K(["AWS account ID", "account ID", "cloud account"],
                    lambda r, x: str(r.randint(10 ** 11, 10 ** 12 - 1)), [
        ("user", "{p} runs in AWS account {v}."),
        ("tool", "infra/{p}.tf:3: account_id = \"{v}\"")]),
    "demo_date": K(["demo date", "review date", "client demo"],
                   lambda r, x: f"{_day(r):%d %b %Y}".lstrip("0"), [
        ("user", "The {p} demo is on {v}."),
        ("user", "{p} has to be demo-ready for {v}")]),
    "namespace": K(["namespace", "k8s namespace", "kube namespace"],
                   lambda r, x: f"{x['p']}-{r.choice(['dev', 'stg', 'prod', 'qa', 'perf'])}", [
        ("tool", "k8s/{p}/kustomization.yaml:2: namespace: {v}"),
        ("user", "deploy {p} into {v}")]),
    "max_upload": K(["max upload size", "upload limit", "file size limit"],
                    lambda r, x: f"{r.choice([5, 8, 10, 16, 20, 25, 32, 50, 64, 100, 200, 500])}MB", [
        ("user", "{p} rejects files above {v}."),
        ("user", "cap {p} uploads at {v}")]),
    "batch_size": K(["batch size", "insert batch size", "chunk size"],
                    lambda r, x: str(r.choice([128, 256, 384, 500, 512, 768, 1000, 1024, 2000, 2048, 4096])), [
        ("user", "{p} should write in batches of {v}."),
        ("tool", "src/{p}/jobs.py:14: BATCH_SIZE = {v}")]),
    "license": K(["license", "licence", "project license"],
                 lambda r, x: r.choice(["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0", "MPL-2.0", "AGPL-3.0"]), [
        ("user", "{p} ships under {v}."),
        ("assistant", "The {p} LICENSE file is {v}.")]),
    "reviewer": K(["reviewer", "approver", "required reviewer"], lambda r, x: r.choice(PEOPLE), [
        ("user", "{v} has to approve every {p} release."),
        ("user", "tag {v} on the {p} PR")]),
    "log_level": K(["log level", "logging level", "LOG_LEVEL"],
                   lambda r, x: r.choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), [
        ("tool", "deploy/{p}.env:5: LOG_LEVEL={v}"),
        ("user", "run {p} at {v} in prod")]),
    "db_password": K(["DB password", "database password", "Postgres password"], None, [
        ("user", "{p} connects to its database with password {v}."),
        ("tool", "deploy/{p}.env:9: DB_PASSWORD={v}")], secret=True),
    "backup_time": K(["backup time", "nightly backup", "backup schedule"],
                     lambda r, x: f"{r.randint(0, 5):02d}:{r.choice([0, 15, 30, 45]):02d} UTC", [
        ("user", "{p} backups run at {v}."),
        ("user", "don't deploy {p} near {v}, that's when the backup runs")]),
    "domain": K(["public domain", "API domain", "public hostname"],
                lambda r, x: f"{r.choice(['api', 'app', 'edge'])}.{x['p']}-{r.choice(['cloud', 'labs', 'hq'])}.in", [
        ("user", "{p} is served from {v}."),
        ("tool", "deploy/{p}.env:2: PUBLIC_HOST={v}")]),
    "ip": K(["database IP", "DB host IP", "private IP"],
            lambda r, x: f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(2, 254)}", [
        ("user", "{p}'s database is at {v}."),
        ("tool", "deploy/{p}.env:4: DB_HOST={v}")]),
    "email": K(["support email", "escalation email", "contact email"],
               lambda r, x: f"{r.choice(PEOPLE).lower()}@{x['p']}-team.in", [
        ("user", "{p} escalations go to {v}."),
        ("user", "mail {v} if {p} goes down")]),
    # --- v2 train kinds: new value shapes ---
    "env_name": K(["env var", "environment variable", "config key"],
                  lambda r, x: f"{x['p'].upper()}_{r.choice(['DB', 'CACHE', 'QUEUE', 'API', 'S3'])}_"
                               f"{r.choice(['URL', 'TIMEOUT', 'MODE', 'PATH', 'REGION'])}", [
        ("user", "{p} reads its settings from {v}, nothing else."),
        ("tool", "src/{p}/settings.py:3: os.environ[\"{v}\"]"),
        ("user", "export {v} before starting {p}")]),
    "metric": K(["alert metric", "metric name", "SLO metric"],
                lambda r, x: f"{x['p']}_{r.choice(['request', 'queue', 'job', 'login'])}_"
                             f"{r.choice(['latency_p99', 'errors_total', 'lag_seconds', 'failures'])}", [
        ("user", "Page on {v} for {p}, that's the one that matters."),
        ("tool", "alerts/{p}.rules:12: expr: {v} > 0"),
        ("user", "graph {v} on the {p} dashboard")]),
    "rfc": K(["design doc", "RFC", "spec"], lambda r, x: f"RFC-{r.randint(1, 999):03d}", [
        ("user", "The {p} rewrite is described in {v}."),
        ("assistant", "{p}'s API change follows {v}."),
        ("user", "read {v} before touching {p}")]),
    "git_tag": K(["release tag", "git tag", "release candidate"],
                 lambda r, x: f"v{r.randint(1, 9)}.{r.randint(0, 30)}.{r.randint(0, 9)}-rc{r.randint(1, 5)}", [
        ("user", "Ship {p} from tag {v}."),
        ("tool", "git tag --list {p}: {v}"),
        ("user", "{p} QA signed off on {v}")]),
    "dashboard": K(["dashboard URL", "Grafana link", "dashboard"],
                   lambda r, x: f"https://grafana.{x['p']}-ops.in/d/{_alnum(r, 6)}", [
        ("user", "{p} graphs live at {v}."),
        ("user", "bookmark {v} for {p} on-call")]),
    "cidr": K(["VPC CIDR", "subnet", "network range"],
              lambda r, x: f"10.{r.randint(0, 255)}.0.0/{r.choice([16, 20, 24])}", [
        ("user", "{p} gets the range {v}."),
        ("tool", "infra/{p}.tf:8: cidr_block = \"{v}\""),
        ("user", "carve {p} out of {v}")]),
    "threshold": K(["error threshold", "alert threshold", "error budget"],
                   lambda r, x: f"{r.choice([0.5, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10])}%", [
        ("user", "Alert when {p} errors go above {v}."),
        ("user", "{p} can burn {v} before we freeze")]),
    "usd_limit": K(["spend limit", "invoice limit", "card limit"],
                   lambda r, x: f"${r.randint(1, 99)},{r.randint(0, 9)}00", [
        ("user", "{p} can't spend more than {v} on the vendor."),
        ("user", "cap {p} at {v}, finance said so")]),
    "cron": K(["cron schedule", "job schedule", "schedule"],
              lambda r, x: r.choice([f"0 */{r.randint(2, 12)} * * *", f"{r.choice([0, 15, 30, 45])} {r.randint(0, 23)} * * *",
                                     f"0 {r.randint(0, 23)} * * {r.randint(1, 5)}"]), [
        ("tool", "k8s/{p}/cronjob.yaml:9: schedule: \"{v}\""),
        ("user", "{p} sync runs on {v}")]),
    "window": K(["maintenance window", "downtime window", "deploy window"],
                lambda r, x: f"{r.randint(1, 5)}h{r.choice([0, 15, 30, 45]):02d}m", [
        ("user", "{p} needs a {v} window for the migration."),
        ("user", "budget {v} of downtime for {p}")]),
    "standup": K(["standup time", "sync meeting", "weekly sync"],
                 lambda r, x: f"{r.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])} {r.randint(9, 18)}:{r.choice([0, 15, 30, 45]):02d} IST", [
        ("user", "The {p} sync is {v}."),
        ("user", "{p} folks meet {v}, join if you can")]),
    # --- test-only kinds ---
    "ticket": K(["tracking ticket", "ticket", "Jira issue"],
                lambda r, x: f"{r.choice(['OPS', 'BUG', 'INC'])}-{r.randint(100, 9999)}", [
        ("user", "The {p} outage is tracked in {v}."),
        ("user", "put {v} in the commit message for {p}")], test=True),
    "env_var": K(["feature flag", "flag", "toggle"],
                 lambda r, x: f"{x['p'].upper()}_{r.choice(WORDS).upper()}_ENABLED", [
        ("user", "Turn on {v} to enable the new {p} path."),
        ("tool", "deploy/{p}.env:7: {v}=1")], test=True),
    "webhook": K(["webhook secret", "webhook token", "signing secret"], None, [
        ("user", "{p} signs its webhooks with {v}."),
        ("user", "verify {p} callbacks using {v}")], secret=True, test=True),
    "region": K(["deploy region", "cloud region", "AWS region"],
                lambda r, x: r.choice(["ap-south-1", "ap-south-2", "eu-west-2", "us-east-1", "us-west-2",
                                       "ap-southeast-1", "eu-central-1", "me-central-1"]), [
        ("user", "{p} is deployed in {v}."),
        ("user", "move {p} to {v} for latency")], test=True),
    "commit": K(["last good commit", "rollback commit", "known-good commit"],
                lambda r, x: "".join(r.choices("0123456789abcdef", k=7)), [
        ("user", "If {p} breaks, revert to {v}."),
        ("tool", "git log {p}: {v} fix: pin worker count")], test=True),
    "cluster": K(["k8s cluster", "cluster name", "GKE cluster"],
                 lambda r, x: f"{r.choice(['prod', 'stg', 'shared'])}-{r.choice(['blr', 'bom', 'fra', 'iad'])}-{r.randint(1, 9)}", [
        ("user", "{p} runs on the {v} cluster."),
        ("tool", "kubectl config current-context: {v}")], test=True),
    "budget": K(["monthly budget", "cloud budget", "spend cap"],
                lambda r, x: f"₹{r.randint(5, 95)},{r.randint(0, 999):03d}", [
        ("user", "{p} must stay under {v} a month."),
        ("user", "finance approved {v} per month for {p}")], test=True),
    "error_code": K(["error code", "failing error code", "error ID"], lambda r, x: f"E{r.randint(1000, 9999)}", [
        ("tool", "logs/{p}.log:88: request failed with {v}"),
        ("user", "{p} keeps throwing {v} on login")], test=True),
    "cache_ttl": K(["cache TTL", "TTL", "cache expiry"], lambda r, x: f"{r.randint(2, 240)}m", [
        ("user", "{p} cache entries live for {v}."),
        ("tool", "src/{p}/cache.py:6: TTL = \"{v}\"")], test=True),
    "sla": K(["uptime SLA", "SLA target", "availability target"],
             lambda r, x: r.choice(["99.5%", "99.9%", "99.95%", "99.99%", "99.0%", "98.5%"]), [
        ("user", "{p} promises {v} uptime to customers."),
        ("user", "{p} is contractually at {v}")], changeable=False, test=True),
    "embed_model": K(["embedding model", "embedder", "encoder model"],
                     lambda r, x: r.choice(["bge-small-en", "bge-m3", "e5-base-v2", "gte-large", "nomic-embed-text",
                                            "all-MiniLM-L6-v2"]), [
        ("user", "{p} embeds documents with {v}."),
        ("assistant", "{p} search uses {v} for vectors.")], test=True),
}
TEST_KINDS = {k for k, s in KINDS.items() if s["test"]}

# Shared by every kind. Last one held out.
GENERIC = [
    ("user", "For {p}, the {label} is {v}."),
    ("user", "{p} {label}: {v}"),
    ("assistant", "Confirmed: the {label} for {p} is {v}."),
    ("user", "note: {p} uses {v} as its {label}"),
    ("assistant", "From the docs, the {p} {label} is {v}."),
    ("tool", "{p}/{file}:{n}: {key} = {v}"),
    ("user", "set the {label} of {p} to {v} please"),
    ("assistant", "I'll use {v} for the {p} {label}."),
    ("tool", "config/{p}.yaml:{n}:   {key_lower}: {v}"),
    ("user", "{p} {label} -> {v}"),
    ("user", "- {label} ({p}): {v}"),
    ("tool", "config/{p}.json:{n}:   \"{key_lower}\": \"{v}\","),
    ("assistant", "So far: {p} has its {label} set to {v}."),
    ("user", "{p}.{key_lower} = {v}"),
    ("user", "remember {v}, that's the {label} on {p}"),
]
GENERIC_TEST_ONLY = [  # whole styles never seen in training
    ("user", "Q: {p} {label}? A: {v}"),
    ("tool", "{p}/pyproject.toml:{n}: {key_lower} = \"{v}\""),
    ("assistant", "{v} ({p} {label}) is confirmed."),
]
UPDATES = [
    ("user", "Correction: the {label} for {p} is now {v} (it was {old})."),
    ("user", "Update: {p} {label} changed to {v}."),
    ("assistant", "Heads up, the {p} {label} moved from {old} to {v}."),
    ("user", "{p}: switch the {label} to {v}, {old} is retired"),
    ("tool", "config/{p}.yaml:{n}: -  {key_lower}: {old}\nconfig/{p}.yaml:{n}: +  {key_lower}: {v}"),
    ("user", "{p} {label}: {old} -> {v}"),
    ("user", "scratch that, {p} {label} should be {v}"),
]
# Value-less mentions: the kind is named, no value is given. Last one held out.
OPEN = [
    [("user", "We still need to pick the {label} for {p}.")],
    [("user", "What's the {label} for {p} again?"),
     ("assistant", "I can't find it in this chat or the repo; someone needs to confirm it.")],
    [("assistant", "I couldn't find the {p} {label} anywhere in the repo.")],
    [("user", "{p} will need a {label} eventually, not decided yet.")],
    [("user", "don't guess the {p} {label}, ask {person} first")],
]
# Question now, answer (the fact statement) later. Last one held out.
ASK = [("user", "What's the {label} for {p}? I forgot."),
       ("user", "Does anyone know the {p} {label}?"),
       ("assistant", "I'll need the {p} {label} before the next step."),
       ("user", "can you dig up the {label} for {p}?")]
# Rejected proposals. {d} must not reach the summary. Last of each list held out.
REJECT_KEEP = [  # a fact exists: the proposal is turned down and {v} stays
    [("user", "Should we change the {p} {label} to {d}?"), ("assistant", "No, keep {v}; {d} was never agreed.")],
    [("assistant", "I could set the {p} {label} to {d}."), ("user", "no, leave it at {v}")],
    [("user", "the wiki still says {d} for the {p} {label}; that's outdated, it's {v}")],
    [("assistant", "Earlier drafts used {d} as the {p} {label}; the current one is {v}.")],
    [("user", "what about {d} for the {p} {label}?"), ("user", "never mind, {v} stays")],
]
REJECT_OPEN = [  # no fact: the idea is dropped and the item stays open
    [("user", "Maybe {d} for the {p} {label}?"), ("assistant", "Let's not decide that yet; {d} is only an idea.")],
    [("assistant", "One option for the {p} {label} is {d}, but nothing is decided.")],
    [("assistant", "For example, something like {d} would work as the {p} {label}, but nothing is set yet.")],
    [("user", "{d} as the {p} {label}? no, too early to say")],
]

# Filler: (question, answer, Context note). Last one held out.
CHAT = [
    ("Why does the {p} test hang on CI?", "It waits on a socket that CI never opens; I added a timeout.",
     "{p}: CI test hung on a socket CI never opens; timeout added."),
    ("Can you explain what the retry decorator in {p} does?", "It retries the call up to three times with backoff and re-raises the last error.",
     "{p}: retry decorator retries 3 times with backoff, then re-raises."),
    ("Is the {p} cache safe to clear?", "Yes, it rebuilds on the next request; only the first call gets slower.",
     "{p}: cache is safe to clear; it rebuilds on the next request."),
    ("Rename the helper in {p} to something clearer.", "Renamed it and updated the two call sites.",
     "{p}: helper renamed, both call sites updated."),
    ("Why is the {p} build so slow?", "The Docker layer cache is invalidated by a copied lockfile; I reordered the steps.",
     "{p}: build was slow from cache invalidation; Dockerfile steps reordered."),
    ("Can we drop the legacy endpoint in {p}?", "Not yet, two clients still call it; I added a deprecation warning.",
     "{p}: legacy endpoint stays for now (2 clients); deprecation warning added."),
    ("Is the flaky {p} test fixed?", "Yes, it depended on dict order; it sorts the keys now.",
     "{p}: flaky test fixed (sorted keys)."),
    ("What's left on the {p} refactor?", "Only the error messages; the logic is done.",
     "{p}: refactor done except the error messages."),
]
ACKS = [("user", "ok thanks"), ("assistant", "Done."), ("user", "looks good, continue"), ("assistant", "On it."),
        ("user", "nice"), ("assistant", "Moving on.")]


def _key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()


class Gen:
    def __init__(self, split: str, i: int, attempt: int):
        self.rng = random.Random(f"delbina-sum-{VERSION}-{split}-{i}-{attempt}")
        self.test = split == "test"
        self.taken: list[str] = []  # every value and distractor in this sample

    def pick(self, items: list):
        return self.rng.choice(items if self.test or len(items) == 1 else items[:-1])

    def value(self, spec: dict, extra: dict) -> str | None:
        """A value that neither equals nor is a whole token of / contains any value already used."""
        for _ in range(30):
            v = spec["value"](self.rng, extra)
            if all(v != t and not _has(t, v) and not _has(v, t) for t in self.taken):
                self.taken.append(v)
                return v
        return None

    # --- filler ---
    def tool_step(self, p: str, buried: str | None = None) -> list[str]:
        r = self.rng
        tool, args = r.choice([("grep", {"pattern": f"{p}_{r.choice(WORDS)}"}),
                               ("read_file", {"path": f"src/{p}/{r.choice(['core', 'util', 'api'])}.py"}),
                               ("run_tests", {"path": f"tests/test_{p}.py"})])
        rows = [f"src/{p}/{r.choice(['core', 'util', 'api'])}.py:{r.randint(1, 99)}: "
                f"{r.choice(['def', 'return', 'raise', 'log'])} {r.choice(WORDS)}_{r.choice(WORDS)}"
                for _ in range(r.randint(4, 18))]
        if buried:
            rows.insert(r.randint(0, len(rows)), buried)
        if r.random() < 0.3:
            rows.insert(len(rows) // 2, TRIM_MARK.format(n=r.randint(200, 3000)))
        out = [f"assistant: {tool}({json.dumps(args)})", "tool: " + "\n".join(rows)]
        if r.random() < 0.6:
            out.append(f"assistant: Checked {p}, nothing unusual there.")
        return out

    def filler(self, projects: list[str]) -> tuple[list[str], str | None]:
        r, p = self.rng, self.rng.choice(projects)
        roll = r.random()
        if roll < 0.5:
            return [f"user: Look into the {p} module."] + self.tool_step(p), None
        if roll < 0.82:
            q, a, note = self.pick(CHAT)
            return [f"user: {q.format(p=p)}", f"assistant: {a}"], note.format(p=p)
        role, text = r.choice(ACKS)
        return [f"{role}: {text}"], None


def _label(spec: dict, gen: Gen, extra: dict) -> str:
    return gen.pick(spec["labels"]).format(**extra)


def _statement(gen: Gen, kind: str, f: dict, v: str) -> list[str]:
    spec = KINDS[kind]
    tpls = spec["say"][:-1] + GENERIC[:-1] if not gen.test else spec["say"] + GENERIC + GENERIC_TEST_ONLY
    role, text = gen.rng.choice(tpls)
    f["how"] = "test_style" if (role, text) in GENERIC_TEST_ONLY else role
    ctx = {**f["extra"], "p": f["subject"], "v": v, "label": f["label"], "key": _key(f["label"]),
           "key_lower": _key(f["label"]).lower(), "file": gen.rng.choice(["settings.py", ".env", "config.py"]),
           "n": gen.rng.randint(1, 99)}
    text = text.format(**ctx)
    return gen.tool_step(f["subject"], buried=text) if role == "tool" else [f"{role}: {text}"]


def _pairs(tpl: list, **ctx) -> list[str]:
    return [f"{role}: {text.format(**ctx)}" for role, text in tpl]


def _build(split: str, i: int, attempt: int) -> dict:
    gen = Gen(split, i, attempt)
    r = gen.rng
    kinds = [k for k, s in KINDS.items() if gen.test or not s["test"]]
    projects = r.sample(TEST_PROJECTS if gen.test else TRAIN_PROJECTS, r.randint(2, 4))
    used: set[tuple[str, str]] = set()

    def new_item(kind: str, p: str) -> dict:
        spec = KINDS[kind]
        extra = {"p": p, **(spec["extra"](r) if spec["extra"] else {})}
        return {"kind": kind, "subject": p, "extra": extra, "label": _label(spec, gen, extra), "stale": []}

    # facts
    facts, secrets, n_facts = [], 0, r.randint(2, 9)
    for _ in range(200):
        if len(facts) >= n_facts:
            break
        kind, p = r.choice(kinds), r.choice(projects)
        if (kind, p) in used:
            continue
        f = new_item(kind, p)
        if KINDS[kind]["secret"]:
            secrets += 1
            f["value"] = f"[[SECRET_{secrets}]]"
        else:
            f["value"] = gen.value(KINDS[kind], f["extra"])
            if f["value"] is None:
                continue
        used.add((kind, p))
        facts.append(f)

    # changes: the newest value counts, older ones are stale
    for f in facts:
        if KINDS[f["kind"]]["changeable"] and r.random() < 0.3:
            new = gen.value(KINDS[f["kind"]], f["extra"])
            if new:
                f["stale"].append(f["value"])
                f["value"] = new

    # open items (kind named, no value) and rejected proposals
    opens: list[dict] = []
    for _ in range(r.choices([0, 1, 2], [0.4, 0.4, 0.2])[0]):
        kind, p = r.choice(kinds), r.choice(projects)
        if (kind, p) not in used:
            used.add((kind, p))
            opens.append(new_item(kind, p))
    distractors: list[str] = []
    rejects: list[tuple[str, dict, str]] = []  # ("keep" | "open", item, d)
    for _ in range(r.choices([0, 1, 2], [0.4, 0.4, 0.2])[0]):
        plain = [f for f in facts if not KINDS[f["kind"]]["secret"] and not f["stale"]
                 and not any(f is it for _, it, _ in rejects)]
        if plain and r.random() < 0.6:
            f = r.choice(plain)
            d = gen.value(KINDS[f["kind"]], f["extra"])
            if d:
                rejects.append(("keep", f, d))
        elif len(opens) < MAX_OPEN:
            kind, p = r.choice([k for k in kinds if not KINDS[k]["secret"]]), r.choice(projects)
            if (kind, p) not in used:
                o = new_item(kind, p)
                d = gen.value(KINDS[kind], o["extra"])
                if d:
                    used.add((kind, p))
                    opens.append(o)
                    rejects.append(("open", o, d))
    distractors = [d for _, _, d in rejects]

    # previous summary: some facts arrive already summarized
    prev = [f for f in facts if r.random() < 0.3] if r.random() < 0.35 else []
    prev_notes: list[str] = []
    prev_line = None
    if prev:
        first = {id(f): f["stale"][0] if f["stale"] else f["value"] for f in prev}
        for f in prev:
            f["where"] = "previous"
        if r.random() < 0.4:  # v0 prose style
            prev_line = (f"previous summary: The work so far covered {', '.join(projects)}. Known facts: "
                         + "; ".join(f"{f['subject']} {f['label']}: {first[id(f)]}" for f in prev) + ".")
        else:  # the model's own output format, as delbina feeds it back
            body = ["Facts:"] + [f"- {f['subject']}: {f['label']}: {first[id(f)]}" for f in prev]
            if r.random() < 0.5:
                q, a, note = gen.pick(CHAT)
                prev_notes.append(note.format(p=r.choice(projects)))
                body += ["Context:", f"- {prev_notes[0]}"]
            prev_line = "previous summary: " + "\n".join(body)

    # schedule events over the filler steps
    steps = r.randint(4, 28)
    events: dict[int, list[tuple]] = {}
    add = lambda at, ev: events.setdefault(at, []).append(ev)
    for f in facts:
        at = -1
        if f.get("where") != "previous":
            at = r.randrange(steps)
            if at > 0 and r.random() < 0.2:
                add(r.randrange(at), ("ask", f))
            add(at, ("say", f))
            f["where"] = ("early", "middle", "late")[min(at * 3 // steps, 2)]
        if f["stale"]:
            add(r.randint(max(at + 1, 0), steps), ("update", f))
        f["_at"] = at
    for o in opens:
        if not any(o is item for _, item, _ in rejects):
            add(r.randrange(steps + 1), ("open", o))
    for how, item, d in rejects:
        at = r.randint(max(item.get("_at", -1) + 1, 0), steps) if how == "keep" else r.randrange(steps + 1)
        add(at, ("reject_" + how, item, d))

    lines = [prev_line] if prev_line else []
    planted: set[int] = set()  # indexes of lines that may carry values
    notes: list[str] = list(prev_notes)
    for s in range(steps + 1):
        for ev in events.get(s, []):
            start = len(lines)
            action, f = ev[0], ev[1]
            ctx = {**f["extra"], "p": f["subject"], "label": f["label"]}
            if action == "say":
                lines += _statement(gen, f["kind"], f, f["stale"][0] if f["stale"] else f["value"])
            elif action == "update":
                role, text = gen.pick(UPDATES)
                text = text.format(v=f["value"], old=f["stale"][0], key_lower=_key(f["label"]).lower(),
                                   n=r.randint(1, 99), **ctx)
                lines += gen.tool_step(f["subject"], buried=text) if role == "tool" else [f"{role}: {text}"]
            elif action == "ask":
                role, text = gen.pick(ASK)
                lines.append(f"{role}: " + text.format(**ctx))
            elif action == "open":
                lines += _pairs(gen.pick(OPEN), person=r.choice(PEOPLE), **ctx)
            elif action == "reject_keep":
                lines += _pairs(gen.pick(REJECT_KEEP), v=f["value"], d=ev[2], **ctx)
            elif action == "reject_open":
                lines += _pairs(gen.pick(REJECT_OPEN), d=ev[2], **ctx)
            planted.update(range(start, len(lines)))
            if r.random() < 0.4:
                lines.append(r.choice(["assistant: Noted.", "assistant: Got it.", "assistant: OK."]))
        if s < steps:
            more, note = gen.filler(projects)
            lines += more
            if note and note not in notes:
                notes.append(note)
    if prev_line:
        planted.add(0)

    # filler must not contain any planted value or distractor
    for j, line in enumerate(lines):
        if j not in planted:
            for v in gen.taken:
                assert not _has(line, v), "filler collides with a value"

    out_facts = [{"kind": f["kind"], "subject": f["subject"], "value": f["value"], "stale": f["stale"],
                  "where": f["where"], "how": f.get("how", "previous"), "label": f["label"]} for f in facts]
    out_open = [{"kind": o["kind"], "subject": o["subject"], "label": o["label"]} for o in opens]
    sample = {
        "id": f"{VERSION}-{split}-{i:06d}",
        "split": split,
        "lines": lines,
        "facts": out_facts,
        "open": out_open,
        "distractors": distractors,
        "n_chars": len("\n".join(lines)),
        "has_previous": bool(prev_line),
        "source": "synthetic",
        "generator": f"delbina datagen {VERSION}",
        "license": "Apache-2.0",
    }
    sample["target"] = target_summary(lines, out_facts, out_open, notes)
    verify(sample)
    return sample


RETRY = ("filler collides", "redaction changed", "target collides")  # resample; anything else is a bug


def make_sample(split: str, i: int) -> dict:
    for attempt in range(20):  # deterministic: same (split, i) always lands on the same attempt
        try:
            return _build(split, i, attempt)
        except AssertionError as e:
            if not str(e).startswith(RETRY):
                raise
    raise RuntimeError(f"{split} {i}: no clean sample in 20 attempts")


def open_note(o: dict) -> str:
    return f"{o['subject']}: {o['label']} not known yet."


def target_summary(lines: list[str], facts: list[dict], opens: list[dict], notes: list[str]) -> str:
    """Facts in order of first mention, newest value only. Context: open items first
    (at most MAX_OPEN), then the latest notes, MAX_NOTES lines in all."""
    text = "\n".join(lines)
    order = sorted(facts, key=lambda f: min(text.find(v) for v in [f["value"], *f["stale"]]))
    out = ["Facts:"] + [f"- {f['subject']}: {f['label']}: {f['value']}" for f in order]
    ctx = [open_note(o) for o in opens[:MAX_OPEN]]
    ctx += notes[-(MAX_NOTES - len(ctx)):] if MAX_NOTES > len(ctx) else []
    if ctx:
        out += ["Context:"] + [f"- {n}" for n in ctx]
    return "\n".join(out)


def verify(sample: dict) -> None:
    text = "\n".join(sample["lines"])
    for f in sample["facts"]:
        for v in [f["value"], *f["stale"]]:
            assert v in text, f"{sample['id']}: {v!r} missing from lines"
    for d in sample["distractors"]:
        assert d in text, f"{sample['id']}: distractor {d!r} missing from lines"
    for line in sample["lines"]:
        assert redact_text(line) == line, f"redaction changed {sample['id']}: {line[:80]!r}"
    r = score(sample["target"], sample["facts"], sample["distractors"])
    if not r["missing"] and not r["unbound"]:  # an old value or distractor matched a label or note word
        assert r["ok"], f"target collides {sample['id']}: {r['stale_hits'] + r['distractor_hits']}"
    assert r["ok"], f"{sample['id']}: target not clean: {r}"


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
            s = make_sample("train", i)
            print("\n".join(s["lines"]), "\n\n--- target ---\n" + s["target"], "\n", file=sys.stdout)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"version": VERSION}
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        kinds, sizes, opens, dis, prev = Counter(), [], 0, 0, 0
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(n):
                s = make_sample(split, i)
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                kinds.update(f["kind"] for f in s["facts"])
                sizes.append(s["n_chars"])
                opens += len(s["open"]); dis += len(s["distractors"]); prev += s["has_previous"]
        sizes.sort()
        stats[split] = {"samples": n, "facts": sum(kinds.values()), "open_items": opens, "distractors": dis,
                        "with_previous": prev, "kinds": dict(sorted(kinds.items())),
                        "chars_p50": sizes[n // 2] if n else 0, "chars_p95": sizes[int(n * 0.95)] if n else 0,
                        "chars_max": sizes[-1] if n else 0}
        print(f"{split:5} {n:5} samples  {stats[split]['facts']:6} facts  {len(kinds)} kinds  "
              f"chars p50 {stats[split]['chars_p50']}  p95 {stats[split]['chars_p95']}", file=sys.stderr)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
