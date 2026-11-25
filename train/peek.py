# usage: python train/peek.py <id> <needle>   (case-insensitive)
import json, re, sys
from huggingface_hub import hf_hub_download

rid, needle = sys.argv[1], sys.argv[2].lower()
path = hf_hub_download("delbina/delbina-sum", "eval_test_v0.jsonl", revision="v3")

def walk(x, p=""):
    if isinstance(x, str):
        yield p, x
    elif isinstance(x, dict):
        for k, v in x.items(): yield from walk(v, f"{p}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from walk(v, f"{p}[{i}]")

for line in open(path, encoding="utf-8"):
    row = json.loads(line)
    if rid not in line[:200] and rid not in json.dumps(row.get("id", "")):
        continue
    for p, s in walk(row):
        for m in re.finditer(re.escape(needle), s.lower()):
            a, b = max(0, m.start() - 150), m.end() + 150
            print(f"--- {p}\n...{s[a:b]}...\n")
    break