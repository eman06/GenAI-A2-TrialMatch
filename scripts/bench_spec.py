"""Task 5: greedy speculative decoding vs ordinary greedy (KV-cached) decoding.

Target: fine-tuned SmolLM2-360M (LoRA merged). Drafts: SmolLM2-135M-Instruct
(same 49,152-token BPE tokenizer) - base, and the LoRA-fine-tuned draft if
available. Two workloads on VALIDATION prompts:
  task   : the real screening prompt, stop at <|im_end|> (short JSON answers)
  long   : same prompts, EOS disabled, 64 new tokens (long-generation regime)
For each k we check token equality with plain greedy and record acceptance,
latency, tokens/s and memory. Also runs HF assisted generation once as an
external reference for token equality (not used for our numbers).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from trialmatch.bench_utils import summarize, timed  # noqa: E402
from trialmatch.config import (ADAPTER_DIR, DRAFT_ADAPTER_DIR, DRAFT_MODEL, RESULTS, TARGET_MODEL,  # noqa: E402
                               model_path)
from trialmatch.data import encode_example, load_split, parse_note  # noqa: E402
from trialmatch.generation import greedy_kv, speculative_greedy  # noqa: E402
from trialmatch.model import load_decoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--ks", default="1,2,4,6")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tdir, ddir = str(model_path(TARGET_MODEL)), str(model_path(DRAFT_MODEL))
    tok = AutoTokenizer.from_pretrained(tdir)
    dtok = AutoTokenizer.from_pretrained(ddir)
    assert tok.get_vocab() == dtok.get_vocab(), "draft/target tokenizers differ"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    target = load_decoder(tdir, adapter_path=str(ADAPTER_DIR) if Path(ADAPTER_DIR, "adapter.pt").exists() else None)
    drafts = {"draft_base_135m": load_decoder(ddir)}
    if Path(DRAFT_ADAPTER_DIR, "adapter.pt").exists():
        drafts["draft_lora_135m"] = load_decoder(ddir, adapter_path=str(DRAFT_ADAPTER_DIR))

    df = load_split()["val"]
    # spread prompts over different patients
    picks = df.groupby("patient_id").head(1).head(args.n_prompts)
    prompts = []
    for _, r in picks.iterrows():
        e = encode_example(tok, parse_note(r.note), r.criterion_type, r.criterion_text)
        prompts.append(e.prefix_ids + e.suffix_ids)
    ks = [int(x) for x in args.ks.split(",")]
    out = {"target": TARGET_MODEL, "drafts": list(drafts), "tokenizer_identical": True, "threads": args.threads,
           "repeats": args.repeats, "workloads": {}}
    for wl, (max_new, stop) in {"task": (48, eos), "long": (64, None)}.items():
        rows = []
        for pi, p in enumerate(prompts):
            ts, base, mem = timed(lambda: greedy_kv(target, p, max_new, eos=stop), warmup=1, repeats=args.repeats)
            row = {"prompt": pi, "prompt_tokens": len(p), "new_tokens": len(base.tokens),
                   "greedy": {"total_s": summarize(ts)["median"], "tokens_per_s": len(base.tokens) / summarize(ts)["median"],
                              "peak_rss_delta_mb": mem / 2**20}}
            for dname, dm in drafts.items():
                for k in ks:
                    ts2, sp, mem2 = timed(lambda: speculative_greedy(target, dm, p, max_new, k=k, eos=stop),
                                          warmup=1, repeats=args.repeats)
                    s = sp.stats
                    row[f"{dname}_k{k}"] = {
                        "equal_to_greedy": sp.tokens == base.tokens, "total_s": summarize(ts2)["median"],
                        "tokens_per_s": len(sp.tokens) / summarize(ts2)["median"],
                        "speedup": summarize(ts)["median"] / summarize(ts2)["median"],
                        "acceptance_rate": s["acceptance_rate"], "accepted": s["accepted"], "drafted": s["drafted"],
                        "rounds": s["rounds"], "accept_hist": s["accept_hist"],
                        "tokens_per_target_forward": s["tokens_per_target_forward"],
                        "target_forwards": s["target_forwards"], "draft_forwards": s["draft_forwards"],
                        "peak_rss_delta_mb": mem2 / 2**20}
                    print(wl, pi, dname, k, row[f"{dname}_k{k}"]["equal_to_greedy"],
                          round(s["acceptance_rate"], 3), round(row[f"{dname}_k{k}"]["speedup"], 3), flush=True)
            rows.append(row)
        out["workloads"][wl] = rows
        json.dump(out, open(RESULTS / "bench_spec.json", "w"), indent=2)

    # external reference: HF assisted generation (only token-equality cross-check)
    try:
        from transformers import AutoModelForCausalLM
        hf_t = AutoModelForCausalLM.from_pretrained(tdir, torch_dtype=torch.float32)
        hf_t.load_state_dict({f"model.{k}" if k != "lm_head.weight" else k: v for k, v in target.state_dict().items()}, strict=False)
        hf_d = AutoModelForCausalLM.from_pretrained(ddir, torch_dtype=torch.float32)
        p = prompts[0]
        ref = hf_t.generate(torch.tensor([p]), max_new_tokens=48, do_sample=False, assistant_model=hf_d,
                            eos_token_id=eos, pad_token_id=eos)[0, len(p):].tolist()
        ours = greedy_kv(target, p, 48, eos=eos).tokens
        out["hf_assisted_reference"] = {"equal_to_ours": ref == ours, "hf_tokens": len(ref), "ours_tokens": len(ours)}
        del hf_t, hf_d
    except Exception as e:  # noqa: BLE001
        out["hf_assisted_reference"] = {"error": repr(e)}
    json.dump(out, open(RESULTS / "bench_spec.json", "w"), indent=2)
    print(out["hf_assisted_reference"])


if __name__ == "__main__":
    main()
