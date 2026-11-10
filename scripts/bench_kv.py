"""Task 3: KV-cached vs uncached greedy decoding - correctness and benchmark.

Same model (fine-tuned SmolLM2-360M, LoRA merged, fp32), same real prompts
(built from VALIDATION-split patient prompts, cut to the requested length),
EOS disabled so every run produces exactly `out` tokens. 1 warm-up run,
then `--repeats` timed runs; perf_counter around synchronised calls
(torch.cuda.synchronize when on GPU).

    python scripts/bench_kv.py --repeats 2
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from trialmatch.bench_utils import summarize, timed  # noqa: E402
from trialmatch.config import ADAPTER_DIR, RESULTS, TARGET_MODEL, model_path  # noqa: E402
from trialmatch.data import encode_example, load_split, parse_note  # noqa: E402
from trialmatch.generation import greedy_kv, greedy_no_cache  # noqa: E402
from trialmatch.model import load_decoder  # noqa: E402


def prompt_stream(tok, n_tokens):
    df = load_split()["val"]
    ids = []
    for _, r in df.iterrows():
        e = encode_example(tok, parse_note(r.note), r.criterion_type, r.criterion_text)
        ids += e.prefix_ids + e.suffix_ids
        if len(ids) >= n_tokens:
            break
    return ids[:n_tokens]


@torch.no_grad()
def first_divergence(model, prompt, a, b):
    """Where two greedy outputs diverge, report the target's top-2 logit margin there."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            logits = model(torch.tensor([prompt + a[:i]]), logits_index=slice(-1, None))[0, -1]
            top = torch.topk(logits, 2)
            return {"index": i, "tokens": [x, y], "top2_margin": (top.values[0] - top.values[1]).item()}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--adapter", default=str(ADAPTER_DIR))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    mdir = str(model_path(TARGET_MODEL))
    tok = AutoTokenizer.from_pretrained(mdir)
    adapter = args.adapter if Path(args.adapter, "adapter.pt").exists() else None
    model = load_decoder(mdir, adapter_path=adapter)
    cfg = model.cfg
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 4
    grid = [(64, 32, True), (256, 32, True), (512, 32, True), (128, 64, True),
            (1024, 64, False), (2048, 32, False)]      # (input, output, run uncached reference?)
    rows = []
    for L, N, with_ref in grid:
        prompt = prompt_stream(tok, L)
        res = {"input_tokens": L, "output_tokens": N}
        modes = {"kv_static": lambda: greedy_kv(model, prompt, N, eos=None, cache_kind="static"),
                 "kv_dynamic": lambda: greedy_kv(model, prompt, N, eos=None, cache_kind="dynamic")}
        if with_ref:
            modes["no_cache"] = lambda: greedy_no_cache(model, prompt, N, eos=None)
        outs = {}
        for name, fn in modes.items():
            ts, g, mem = timed(fn, warmup=1 if name != "no_cache" or L <= 128 else 0, repeats=args.repeats)
            outs[name] = g.tokens
            res[name] = {"total_s": summarize(ts), "prefill_s": g.prefill_s, "decode_s": g.decode_s,
                         "tokens_per_s": N / summarize(ts)["median"], "decode_tokens_per_s": g.decode_tokens_per_s,
                         "peak_rss_delta_mb": mem / 2**20, "kv_cache_mb": g.stats.get("kv_cache_bytes", 0) / 2**20}
            print(L, N, name, round(summarize(ts)["median"], 3), "s", flush=True)
        res["kv_cache_analytic_mb"] = kv_bytes_per_token * (L + N) / 2**20
        res["static_eq_dynamic"] = outs["kv_static"] == outs["kv_dynamic"]
        if with_ref:
            res["token_agreement"] = outs["kv_static"] == outs["no_cache"]
            res["n_agree"] = sum(a == b for a, b in zip(outs["kv_static"], outs["no_cache"]))
            res["first_divergence"] = first_divergence(model, prompt, outs["no_cache"], outs["kv_static"])
            res["speedup_total"] = res["no_cache"]["total_s"]["median"] / res["kv_static"]["total_s"]["median"]
        res["sample_output"] = tok.decode(outs["kv_static"][:40])
        rows.append(res)
        json.dump({"model": TARGET_MODEL, "adapter": adapter, "dtype": "float32", "threads": args.threads,
                   "kv_bytes_per_token": kv_bytes_per_token, "repeats": args.repeats, "rows": rows},
                  open(RESULTS / "bench_kv.json", "w"), indent=2)
    print(json.dumps([{k: r.get(k) for k in ("input_tokens", "output_tokens", "token_agreement", "speedup_total")} for r in rows], indent=1))


if __name__ == "__main__":
    main()
