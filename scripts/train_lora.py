"""LoRA fine-tuning of SmolLM2 on the TrialGPT TRAIN split (Task 1).

Objective: causal-LM cross-entropy on the answer tokens only
(``{"label": ..., "evidence": [...]}<|im_end|>``); prompt tokens are context.

CPU-efficiency trick - shared-prefix batching:
every patient has ~16 criteria and the patient note (~300 tokens) is the
same prefix for all of them. For each chunk of criteria of one patient we run
the prefix ONCE (with gradients), expand its KV cache over the chunk, and run
only the short criterion+answer suffixes as a batch. This is mathematically
identical to running every (prefix, suffix) pair separately (causal attention:
the prefix never sees the suffix) but costs P + n*S tokens instead of n*(P+S).

Usage:  python scripts/train_lora.py --model SmolLM2-360M-Instruct --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from trialmatch.config import ADAPTER_DIR, MAX_PROMPT_TOKENS, RESULTS, TARGET_MODEL, model_path  # noqa: E402
from trialmatch.data import encode_example, load_split, parse_note  # noqa: E402
from trialmatch.model import PrefixRecorder, PrefixReuser, load_decoder  # noqa: E402


def build_groups(df, tok, chunk):
    """-> list of (prefix_ids, [(suffix_ids, target_ids, class_key)]) chunks."""
    groups = []
    trunc = 0
    for pid, sub in df.groupby("patient_id", sort=True):
        items, prefix = [], None
        for _, r in sub.iterrows():
            e = encode_example(tok, parse_note(r.note), r.criterion_type, r.criterion_text,
                               r.expert_eligibility, r.expert_sentences, MAX_PROMPT_TOKENS)
            trunc += bool(e.truncated_sentences)
            prefix = e.prefix_ids
            items.append((e.suffix_ids, e.target_ids, f"{r.criterion_type}/{r.expert_eligibility}"))
        groups.append((pid, prefix, items))
    return groups, trunc


def chunks_for_epoch(groups, chunk, rng):
    out = []
    for pid, prefix, items in groups:
        items = items[:]
        rng.shuffle(items)
        for i in range(0, len(items), chunk):
            out.append((prefix, items[i:i + chunk]))
    rng.shuffle(out)
    return out


def chunk_loss(model, prefix, items, weights, device, ckpt=True):
    """Weighted mean CE over answer tokens for one shared-prefix chunk."""
    cache = PrefixRecorder(model.cfg.num_hidden_layers)
    model(torch.tensor([prefix], device=device), cache=cache, return_hidden=True,
          logits_index=slice(0, 0), checkpoint=ckpt)                  # prefix: fill KV only
    B = len(items)
    seqs = [s + t for s, t, _ in items]
    L = max(len(x) for x in seqs)
    ids = torch.full((B, L), 2, dtype=torch.long, device=device)      # pad = <|im_end|>(2), right padded
    for b, x in enumerate(seqs):
        ids[b, :len(x)] = torch.tensor(x)
    h = model(ids, cache=PrefixReuser(cache), return_hidden=True, checkpoint=ckpt)  # (B, L, d)
    loss_sum, wsum, ntok = 0.0, 0.0, 0
    for b, (s, t, key) in enumerate(items):
        pos = torch.arange(len(s) - 1, len(s) + len(t) - 1, device=device)
        logits = F.linear(h[b, pos], model.embed_tokens.weight)
        l = F.cross_entropy(logits, torch.tensor(t, device=device))
        loss_sum = loss_sum + weights[key] * l
        wsum += weights[key]
        ntok += len(t)
    return loss_sum / wsum, ntok


@torch.no_grad()
def eval_loss(model, groups, device, chunk):
    model.eval()
    tot, n = 0.0, 0
    for pid, prefix, items in groups:
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            l, _ = chunk_loss(model, prefix, part, {k: 1.0 for _, _, k in part}, device)
            tot += l.item() * len(part)
            n += len(part)
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=4, help="criteria sharing one prefix per step")
    ap.add_argument("--class_weight_power", type=float, default=0.5,
                    help="example weight = freq^-p (0 = unweighted)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = str(model_path(args.model))
    tok = AutoTokenizer.from_pretrained(mdir)
    sp = load_split()
    train_groups, tr_trunc = build_groups(sp["train"], tok, args.chunk)
    val_groups, va_trunc = build_groups(sp["val"], tok, args.chunk)
    print(f"truncated prompts: train={tr_trunc} val={va_trunc}")

    freq = Counter(f"{r.criterion_type}/{r.expert_eligibility}" for _, r in sp["train"].iterrows())
    weights = {k: (v / len(sp['train'])) ** (-args.class_weight_power) for k, v in freq.items()}
    mean_w = sum(weights[k] * v for k, v in freq.items()) / len(sp["train"])
    weights = {k: w / mean_w for k, w in weights.items()}             # E[w] = 1 over the train set
    print("class weights:", {k: round(v, 2) for k, v in sorted(weights.items())})

    model = load_decoder(mdir, device=device)
    model.add_lora(r=args.r, alpha=args.alpha, dropout=args.dropout)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable); n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    rng = random.Random(args.seed)
    steps_per_epoch = len(chunks_for_epoch(train_groups, args.chunk, random.Random(0)))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / f"train_log_{args.model}.jsonl"
    logf = open(log_path, "w")
    proc = psutil.Process()
    peak_rss = 0
    t_start = time.time()
    v0 = eval_loss(model, val_groups, device, args.chunk)
    print(f"epoch 0 val_loss {v0:.4f}")
    logf.write(json.dumps({"epoch": 0, "step": 0, "val_loss": v0}) + "\n"); logf.flush()
    best, step = v0, 0
    for ep in range(1, args.epochs + 1):
        for prefix, items in chunks_for_epoch(train_groups, args.chunk, rng):
            t0 = time.time()
            loss, ntok = chunk_loss(model, prefix, items, weights, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
            opt.step(); sched.step(); step += 1
            peak_rss = max(peak_rss, proc.memory_info().rss)
            rec = {"epoch": ep, "step": step, "loss": loss.item(), "grad_norm": gn,
                   "lr": sched.get_last_lr()[0], "n_examples": len(items), "answer_tokens": ntok,
                   "sec": round(time.time() - t0, 2), "rss_gb": round(proc.memory_info().rss / 1e9, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % 5 == 0 or step == 1:
                el = time.time() - t_start
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {rec['sec']}s/step "
                      f"rss {rec['rss_gb']}GB elapsed {el/60:.1f}m eta {(total-step)*el/step/60:.1f}m", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        v = eval_loss(model, val_groups, device, args.chunk)
        print(f"epoch {ep} val_loss {v:.4f}", flush=True)
        logf.write(json.dumps({"epoch": ep, "step": step, "val_loss": v}) + "\n"); logf.flush()
        payload = {"config": {"base_model": args.model, "r": args.r, "alpha": args.alpha,
                              "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                              "layers": [0, model.cfg.num_hidden_layers], "epoch": ep, "val_loss": v,
                              "train_args": vars(args)},
                   "state_dict": model.lora_state_dict()}
        torch.save(payload, out / f"adapter_epoch{ep}.pt")
        if v <= best:
            best = v
            torch.save(payload, out / "adapter.pt")
            print(f"  saved best adapter (epoch {ep})")
        if args.max_steps and step >= args.max_steps:
            break
    summary = {"model": args.model, "args": vars(args), "steps": step, "best_val_loss": best,
               "initial_val_loss": v0, "train_minutes": round((time.time() - t_start) / 60, 1),
               "peak_rss_gb": round(peak_rss / 1e9, 2), "trainable_params": n_tr, "total_params": n_all,
               "truncated_prompts": {"train": tr_trunc, "val": va_trunc}, "class_weights": weights}
    json.dump(summary, open(RESULTS / f"train_summary_{args.model}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
