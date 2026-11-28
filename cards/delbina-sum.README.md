---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language: [en]
library_name: transformers
pipeline_tag: text-generation
tags: [summarization, context-compression, lora, delbina]
---

# delbina-sum (v5, 1.5B)

A 1.5B summarizer for [delbina](https://github.com/delbina/delbina), the context-compression layer
for long chats and agent sessions. It compresses old turns while keeping every specific value
(ports, keys, versions, dates, paths, hosts, owners, decisions, `[[SECRET_n]]` placeholders), with
**only the newest value** and **under the right project**, and drops tool noise.

LoRA (r=16, alpha=32, 2 epochs) on Qwen2.5-1.5B-Instruct, merged. The adapter alone is in `adapter/`.
GGUF builds (Q8_0, Q4_K_M) for Ollama / llama.cpp: [delbina/delbina-sum-GGUF](https://huggingface.co/delbina/delbina-sum-GGUF).

## Results

Greedy decoding, delbina's production prompt. 300 chats per set. Test sets use project names, fact
kinds and phrasings never seen in training. The holdout sets come from the same distribution with
new seeds and were scored once, on a model that had already passed every test bar.

| metric (test, v2 set) | Qwen2.5-1.5B-Instruct | delbina-sum |
|---|---|---|
| fact recall (value present) | 22.5 | **98.0** |
| bound recall (value on a line naming its project) | 20.5 | **97.6** |
| recall on fact kinds never seen in training | 23.3 | **97.2** |
| invented values (% of fact lines) | 0.0* | 0.6 |
| chats with a stale (superseded) value | 40.3 | 3.3 |
| chats where a distractor value leaked | 29.1 | 0.6 |
| chats with every fact right | 0.0 | 85.7 |
| summary length, median chars | 454 | 373 |

\* the base model invents nothing because it keeps almost no values at all.

| set | fact recall | unseen-kind recall | invented |
|---|---|---|---|
| test v2 | 98.0 | 97.2 | 0.6 |
| test v0 (older, different generator) | 92.7 | 86.3 | 0.5 |
| holdout v2 | 97.8 | 97.6 | 0.6 |
| holdout v0 | 90.6 | 87.3 | 0.5 |

Release bars (all pass on test and holdout, fp16 and both GGUF quants): recall gain ≥ +15 points,
invented ≤ 2.0%, unseen-kind recall ≥ 70 (≥ 60 on v0), distractor leaks ≤ 5%.

Per-sample test outputs: `eval_test.jsonl`, `eval_test_v0.jsonl`. All numbers: `results.json`, `holdout_results.json`.

## Use with delbina

Easiest is the GGUF through Ollama, see [delbina-sum-GGUF](https://huggingface.co/delbina/delbina-sum-GGUF).
With transformers or any OpenAI-compatible server (vLLM, llama.cpp server, LM Studio):

```python
from delbina import Compressor, openai_summarizer

summarize = openai_summarizer(model="delbina/delbina-sum", base_url="http://localhost:8000/v1", api_key="none")
compressor = Compressor(summarize=summarize)
```

Use greedy decoding (temperature 0) and a context of at least 8192 tokens; that is how it was evaluated.

## Output format

```
Facts:
- <project>: <what it is>: <value>
Context:
- <up to 3 short notes on decisions or open problems>
```

## Training data

2,000 synthetic agent chats (generator v5 in [delbina-sum](https://github.com/delbina/delbina-sum),
dataset: [delbina-sum-data](https://huggingface.co/datasets/delbina/delbina-sum-data)): several projects
per chat, facts said by the user, the assistant or buried in tool output, values that change or get
rejected later, previous summaries to merge, near-miss filler with no values, and `[[SECRET_n]]`
placeholders. Targets are built from the planted facts, so they are correct by construction; no teacher
model. Trained on 2× T4 (fp16 base, fp32 LoRA), effective batch 16, lr 2e-4 cosine, 131 min.

## Limitations

- **Synthetic data only.** Not yet measured on real chats. Content outside the planted fact kinds
  (long explanations, code, nuanced reasoning) may be compressed away.
- **Wrong-project facts:** about 0.4% of fact lines put a real value under the wrong project, most often
  look-alike cluster names; a few facts come out without a project.
- **Rare inventions:** on unfamiliar input it occasionally writes a default-looking value that is not in
  the chat (for example a placeholder database URL with `postgres:password@localhost`).
- **Invented Context notes on very short chats:** on tiny inputs (a handful of messages) it sometimes adds
  Context lines such as "X: not known yet" for things the chat never mentioned. Treat Context lines as
  unverified and ignore them for very short inputs.
- Recall is weaker on the older v0 style (about 87–93) than on v2 (about 97–98).
- English only. Fixed output format. Recall is exact-match, so a paraphrased value counts as a miss.
- Do not rely on it as the only copy of secrets or critical config: delbina keeps originals in mask mode.
