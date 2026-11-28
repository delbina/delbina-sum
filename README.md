# delbina-sum

A small summarizer model for [delbina](https://github.com/delbina/delbina): it compresses old chat and agent turns while keeping the exact facts (IDs, ports, paths, versions, decisions, `[[SECRET_n]]` placeholders) that later turns depend on.

**Current release: v5, 1.5B.** LoRA fine-tune of Qwen2.5-1.5B-Instruct on synthetic agent chats with planted facts (targets correct by construction, no teacher model). Fact recall 98.0 vs 22.5 for the base model, invented values 0.6%, and every release bar passes on the test sets and on a once-only holdout, in fp16 and in both GGUF quants.

- Model: [delbina/delbina-sum](https://huggingface.co/delbina/delbina-sum) (merged weights + LoRA adapter, full results)
- GGUF for Ollama / llama.cpp: [delbina/delbina-sum-GGUF](https://huggingface.co/delbina/delbina-sum-GGUF) (Q4_K_M default, Q8_0)
- Data: [delbina/delbina-sum-data](https://huggingface.co/datasets/delbina/delbina-sum-data)

## History

| version | base | result |
|---|---|---|
| v0–v5 | Qwen2.5-0.5B | best (v5) passed every test bar but failed the holdout: v0 invented rate 2.5% (bar 2.0%) |
| v5-1.5b | Qwen2.5-1.5B | same v5 data; passes every bar on test and holdout; released |

## Layout

- `datagen/`: synthetic dataset generators (v0, v1 frozen; v2 test split; v5 current train/val), prompts and fact scoring
- `train/ddp_train.py`: LoRA training on 1+ GPUs
- `train/holdout.py`: once-only ship check on held-out chats (aggregates only)
- `train/diagnose.py`: error breakdown on a model's saved test outputs
- `train/rescore.py`: re-score a saved `eval_test.jsonl` with the current metrics
- `notebooks/`: training and eval (Kaggle/Colab); `delbina_sum_gguf.ipynb` builds and re-evaluates the GGUF files
- `cards/`: Hugging Face model cards
- `tests/`

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python datagen/make_dataset_v5.py --out data/v5
python -m pytest -q -p no:cacheprovider
```

## License

Apache-2.0. © 2026 Krupakar S Bilichod.
