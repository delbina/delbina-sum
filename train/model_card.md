---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
language: [en]
library_name: transformers
pipeline_tag: text-generation
tags: [summarization, context-compression, lora, delbina]
---

# delbina-sum (v0)

A 0.5B summarizer for [delbina](https://github.com/delbina/delbina), the context-compression layer
for agent chats. It is trained to keep every specific value from a chat (ports, keys, versions,
dates, paths, hosts, owners, decisions, `[[SECRET_n]]` placeholders) with **only the newest value**
and **under the right project**, and to drop tool noise.

LoRA (r={{LORA_R}}, {{EPOCHS}} epochs) on Qwen2.5-0.5B-Instruct, merged. The LoRA adapter alone is in `adapter/`.

## Results

Held-out test split of [delbina-sum-data](https://huggingface.co/datasets/delbina/delbina-sum-data) v0
(300 chats; project names, two fact kinds and one phrasing per fact kind never seen in training).
Greedy decoding, delbina's production prompt.

| metric | Qwen2.5-0.5B-Instruct | delbina-sum |
|---|---|---|
| fact recall (value present) | {{BASE_RECALL}} | **{{TUNED_RECALL}}** |
| bound recall (value on a line naming its project) | {{BASE_BOUND}} | **{{TUNED_BOUND}}** |
| chats with a stale (superseded) value | {{BASE_STALE}} | {{TUNED_STALE}} |
| chats with every fact right | {{BASE_PERFECT}} | {{TUNED_PERFECT}} |
| summary length, median chars | {{BASE_CHARS}} | {{TUNED_CHARS}} |
| recall on unseen fact kinds (env_var / ticket) | {{HELD_BASE}} | {{HELD_TUNED}} |

Fact recall gain: **{{GAIN}} points**. Per-sample outputs: `eval_test.jsonl`; all numbers: `results.json`.

## Use with delbina

Serve it behind any OpenAI-compatible endpoint (vLLM, llama.cpp server, LM Studio), then:

```python
from delbina import Compressor, openai_summarizer

summarize = openai_summarizer(model="delbina/delbina-sum", base_url="http://localhost:8000/v1", api_key="none")
compressor = Compressor(summarize=summarize)  # plus your usual budget options
```

e.g. `vllm serve delbina/delbina-sum --max-model-len 8192`.

## Output format

```
Facts:
- <project>: <what it is>: <value>
Context:
- <up to 3 short notes on decisions or open problems>
```

## Training data

{{N_TRAIN}} synthetic agent chats from `datagen/make_dataset.py` in
[delbina-sum](https://github.com/delbina/delbina-sum): 2-4 projects and 2-8 planted facts per chat, facts
said by the user, the assistant or buried in tool output, values that change later, previous summaries
to merge, and `[[SECRET_n]]` placeholders. Targets are built directly from the planted facts, so they are
correct by construction; no teacher model was used.

## Limitations

- Trained only on synthetic chats and the fact kinds the generator plants; other important content
  (long explanations, code, nuanced reasoning) may be compressed away.
- English only. Fixed output format.
- Recall is measured by exact value match, so a summary that paraphrases a value counts as a miss.
- Do not rely on it as the only copy of secrets or critical config: delbina keeps originals in mask mode.
