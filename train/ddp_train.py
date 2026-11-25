# Copyright 2026 Krupakar S Bilichod
# SPDX-License-Identifier: Apache-2.0
"""LoRA training only, on 1 or more GPUs. Saves the adapter to <out>/adapter.

    torchrun --nproc_per_node 2 train/ddp_train.py --version v2 --out /kaggle/working/out   # Kaggle T4 x2
    python train/ddp_train.py --version v2 --out /content/out                              # one GPU

Effective batch stays 16 whatever the GPU count: per-GPU batch 1 x grad_accum (16 / n_gpus).
Each process rebuilds the (deterministic) dataset itself, so nothing is passed between them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "train")]

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
EFFECTIVE_BATCH = 16
GENERATORS = {"v1": "make_dataset_v1", "v2": "make_dataset_v2", "v3": "make_dataset_v3", "v4": "make_dataset_v4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v4", choices=sorted(GENERATORS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=2000)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    import sft
    md = __import__(GENERATORS[a.version])

    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    assert EFFECTIVE_BATCH % world == 0, f"{world} GPUs don't divide batch {EFFECTIVE_BATCH}"
    torch.cuda.set_device(rank)
    main_proc = rank == 0

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = [md.make_sample("train", i) for i in range(a.n_train)]
    train_ex, dropped = sft.build_examples(tok, rows, a.max_len)
    if main_proc:
        print(f"[ddp_train] {world} GPU(s), grad_accum {EFFECTIVE_BATCH // world}, "
              f"examples {len(train_ex)}, dropped {dropped}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map={"": rank})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if main_proc:
        model.print_trainable_parameters()

    out = Path(a.out)
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=1,
        gradient_accumulation_steps=EFFECTIVE_BATCH // world,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        fp16=True, logging_steps=10, save_strategy="no", report_to="none", group_by_length=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, remove_unused_columns=False, dataloader_num_workers=2, seed=0)
    t = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=train_ex, data_collator=sft.Collator(tok.pad_token_id))
    trainer.train()
    if trainer.is_world_process_zero():
        model.save_pretrained(out / "adapter")
        (out / "train_minutes.txt").write_text(str(round((time.time() - t) / 60)))
        print(f"[ddp_train] trained in {(time.time() - t) / 60:.0f} min, adapter -> {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
