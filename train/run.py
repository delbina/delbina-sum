# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""Train + eval + push on any CUDA machine (the notebook's steps as a script).

    python train/run.py --check          # data + tokenizer + GPU check, no training (~1 min)
    python train/run.py                  # full run: baseline, LoRA, evals, verdict, private push
    python train/run.py --no-push        # same, keep everything local in outputs/run/

Needs HF_TOKEN in the environment (write access) unless --no-push. Run it inside tmux.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN, EPOCHS, LR, LORA_R, LORA_ALPHA, GRAD_ACCUM = 6144, 2, 2e-4, 16, 32, 16
EVAL_BATCH, MAX_NEW_TOKENS = 8, 384
BAR, INVENTED_MAX, UNSEEN_MIN = 15.0, 2.0, 70.0
HF_MODEL_REPO, HF_DATA_REPO = "delbina/delbina-sum", "delbina/delbina-sum-data"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1"])
    ap.add_argument("--check", action="store_true", help="stop after data, tokenizer and GPU checks")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "run"))
    a = ap.parse_args()
    if not a.no_push and not a.check and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set (or pass --no-push)")

    import torch
    import make_dataset as md0
    import make_dataset_v1 as md
    import sft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
    log("gpu:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB",
        "torch", torch.__version__)
    torch.ones(1, device="cuda").add_(1)  # fails here if this torch build lacks kernels for the GPU

    data = {s: [md.make_sample(s, i) for i in range(n)] for s, n in {"train": 2000, "val": 200, "test": 300}.items()}
    test_v0 = [md0.make_sample("test", i) for i in range(300)]
    log("data", {k: len(v) for k, v in data.items()}, "v0 test", len(test_v0))

    tok = AutoTokenizer.from_pretrained(BASE)
    train_ex, dropped = sft.build_examples(tok, data["train"], MAX_LEN)
    lens = sorted(len(e["input_ids"]) for e in train_ex)
    log(f"train examples {len(train_ex)}, dropped {dropped}, tokens p50 {lens[len(lens) // 2]} max {lens[-1]}")
    if a.check:
        log("CHECK OK")
        return

    unseen = md.TEST_KINDS | md0.TEST_KINDS

    def run_eval(model, name, rows):
        t = time.time()
        outs = sft.summarize(model, tok, rows, batch_size=EVAL_BATCH, max_new_tokens=MAX_NEW_TOKENS)
        res = sft.evaluate(rows, outs, unseen)
        log(f"{name}: {time.time() - t:.0f}s", json.dumps({k: v for k, v in res.items() if k != "recall_by_kind"}))
        return outs, res

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
    base_outs, base_res = run_eval(base, "base", data["test"])
    del base
    torch.cuda.empty_cache()

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    out = Path(a.out)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS, learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
        fp16=True, logging_steps=10, save_strategy="no", report_to="none", group_by_length=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=2, seed=0)
    t = time.time()
    Trainer(model=model, args=args, train_dataset=train_ex, data_collator=sft.Collator(tok.pad_token_id)).train()
    train_min = round((time.time() - t) / 60)
    log(f"trained in {train_min} min")

    model.save_pretrained(out / "adapter")
    tuned = model.merge_and_unload().half().eval()
    tuned.gradient_checkpointing_disable()
    tuned.config.use_cache = True
    tuned_outs, tuned_res = run_eval(tuned, "tuned", data["test"])
    _, tuned_v0 = run_eval(tuned, "tuned on v0 test", test_v0)

    gain = tuned_res["fact_recall"] - base_res["fact_recall"]
    checks = {f"recall gain >= +{BAR}": gain >= BAR,
              f"invented <= {INVENTED_MAX}%": tuned_res["invented_rate"] <= INVENTED_MAX,
              f"unseen-kind recall >= {UNSEEN_MIN}": (tuned_res["unseen_recall"] or 0) >= UNSEEN_MIN}
    passed = all(checks.values())
    log(f"gain {gain:+.1f} pts")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log("RELEASE CANDIDATE" if passed else "HOLD")

    mdir = out / "model"
    tuned.save_pretrained(mdir, safe_serialization=True)
    tok.save_pretrained(mdir)
    with open(mdir / "eval_test.jsonl", "w", encoding="utf-8") as fh:
        for r, b, t_ in zip(data["test"], base_outs, tuned_outs):
            fh.write(json.dumps({"id": r["id"], "facts": r["facts"], "base": b["summary"], "tuned": t_["summary"]},
                                ensure_ascii=False) + "\n")
    results = {"version": a.version, "base_model": BASE, "base": base_res, "tuned": tuned_res,
               "tuned_on_v0_test": tuned_v0, "gain_pts": round(gain, 1), "checks": checks, "pass": passed,
               "gpu": torch.cuda.get_device_name(0),
               "train": {"examples": len(train_ex), "dropped": dropped, "epochs": EPOCHS, "lr": LR, "minutes": train_min,
                         "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "effective_batch": GRAD_ACCUM, "max_len": MAX_LEN}}
    (mdir / "results.json").write_text(json.dumps(results, indent=1))
    log("saved", mdir)

    if a.no_push:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(mdir), repo_id=HF_MODEL_REPO,
                      commit_message=f"{a.version}: fact recall {tuned_res['fact_recall']} (base {base_res['fact_recall']}), "
                                     f"{'pass' if passed else 'hold'}")
    api.upload_folder(folder_path=str(out / "adapter"), path_in_repo="adapter", repo_id=HF_MODEL_REPO,
                      commit_message=f"{a.version}: LoRA adapter")
    ddir = out / "data" / a.version
    ddir.mkdir(parents=True, exist_ok=True)
    for s, rows in data.items():
        with open(ddir / f"{s}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    api.create_repo(HF_DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ddir), path_in_repo=a.version, repo_id=HF_DATA_REPO, repo_type="dataset",
                      commit_message=f"{a.version}: synthetic chats + correct-by-construction targets")
    log("pushed (private)")


if __name__ == "__main__":
    main()
