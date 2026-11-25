# delbina-sum

A small summarizer model for [delbina](https://github.com/delbina/delbina): it compresses old chat and agent turns while keeping the exact facts (IDs, ports, paths, versions, decisions, `[[SECRET_n]]` placeholders) that later turns depend on.

Plan: LoRA fine-tune of Qwen2.5-0.5B-Instruct, distilled from Qwen2.5-7B/14B-Instruct on synthetic agent chats with planted facts. The base model's score on a frozen test set is recorded here before any training. Success means at least +15 points of fact recall over the base model at the same summary length. If it falls short, the dataset and the negative result are still published.

## Layout

- `datagen/`: synthetic dataset generators (v0, v1 frozen; v2 current), prompts and fact scoring
- `train/rescore.py`: re-score a saved `eval_test.jsonl` with the current metrics
- `notebooks/`: teacher run, training and eval (Colab/Kaggle)
- `tests/`

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python datagen/make_dataset.py          # writes data/v0/{train,val,test}.jsonl (frozen)
python datagen/make_dataset_v1.py       # writes data/v1: 43 fact kinds, open items, rejected values
python datagen/make_dataset_v2.py       # writes data/v2: + new value shapes, styles, diff updates, more rejects
python -m pytest -q -p no:cacheprovider
```

## License

Apache-2.0. © 2026 Krupakar S Bilichod.
