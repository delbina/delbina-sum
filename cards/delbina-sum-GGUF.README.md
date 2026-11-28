---
license: apache-2.0
base_model: delbina/delbina-sum
language: [en]
pipeline_tag: text-generation
tags: [summarization, context-compression, gguf, ollama, delbina]
---

# delbina-sum GGUF (v5, 1.5B)

GGUF builds of [delbina/delbina-sum](https://huggingface.co/delbina/delbina-sum), the summarizer for
[delbina](https://github.com/delbina/delbina). Converted and quantized with llama.cpp from the merged
model at revision `d33bbb2`; each file was re-evaluated with the same metrics and passes every release bar.

| file | size | test recall (v2 / v0) | holdout recall (v2 / v0) | unseen-kind recall, holdout v0 | invented (holdout v2 / v0) |
|---|---|---|---|---|---|
| fp16 reference (safetensors in the [main repo](https://huggingface.co/delbina/delbina-sum)) | 2.9 GB | 98.0 / 92.7 | 97.8 / 90.6 | 87.3 | 0.6 / 0.5 |
| `delbina-sum-1.5b-Q8_0.gguf` | 1.6 GB | 98.5 / 92.9 | 98.1 / 91.6 | 85.5 | 0.4 / 0.3 |
| `delbina-sum-1.5b-Q4_K_M.gguf` (default) | 941 MB | 98.2 / 91.9 | 97.2 / 91.0 | 83.6 | 0.3 / 0.4 |

Q8_0 matches fp16. Q4_K_M loses about 4 points of recall on fact kinds never seen in training
(v0 style); everything else holds. All numbers: `gguf_results.json`.

## Use with delbina and Ollama

```bash
ollama pull hf.co/delbina/delbina-sum-GGUF:Q4_K_M
```

```python
from delbina import Compressor, ollama_summarizer

summarize = ollama_summarizer(model="hf.co/delbina/delbina-sum-GGUF:Q4_K_M")
compressor = Compressor(summarize=summarize)
```

It was evaluated with greedy decoding (temperature 0) and an 8192-token context. Ollama's defaults
are different, so use delbina ≥ 0.5.0, which sets both.

See the [main model card](https://huggingface.co/delbina/delbina-sum) for training data, output
format and limitations.

Can invent "not known yet" Context notes on very short chats; see the main model card.
