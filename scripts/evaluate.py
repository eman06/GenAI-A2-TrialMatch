"""Criterion-level evaluation of a (base or fine-tuned) model on val or test.

Decoding: deterministic greedy, max_new_tokens=48, fp32, SDPA attention,
KV cache with the patient-note prefix computed once per patient.

    python scripts/evaluate.py --split val  --tag base      --adapter none
    python scripts/evaluate.py --split val  --tag finetuned
    python scripts/evaluate.py --split test --tag finetuned --final   # only after model selection
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import ADAPTER_DIR, RESULTS, TARGET_MODEL  # noqa: E402
from trialmatch.data import load_split, parse_note  # noqa: E402
from trialmatch.metrics import full_report  # noqa: E402
from trialmatch.pipeline import Screener, ScreenerConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--adapter", default=str(ADAPTER_DIR), help="'none' for the base model")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("refusing to evaluate on TEST without --final (test is reserved for the final run)")

    adapter = None if args.adapter == "none" else args.adapter
    scr = Screener(ScreenerConfig(target=args.model, adapter=adapter, draft=None, max_new_tokens=args.max_new_tokens,
                                  label_strategy="generated"))
    if adapter and scr.adapter_used is None:
        sys.exit(f"adapter not found in {adapter}")
    df = load_split()[args.split]
    if args.limit:
        df = df.head(args.limit)
    out_dir = RESULTS / "eval"; out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"preds_{args.tag}_{args.split}.jsonl"
    rows = []
    t0 = time.time()
    with open(pred_path, "w") as f:
        for pid, sub in df.groupby("patient_id", sort=True):
            sents = parse_note(sub.iloc[0].note)
            cache = scr.prefill_patient(sents)
            for _, r in sub.iterrows():
                res = scr.screen_criterion(sents, r.criterion_type, r.criterion_text, "kv", cache)
                row = {"annotation_id": int(r.annotation_id), "patient_id": pid, "trial_id": r.trial_id,
                       "criterion_type": r.criterion_type, "criterion": r.criterion_text,
                       "gold_label": r.expert_eligibility, "pred_label": res["generated_label"],
                       "label_logprobs": res["label_logprobs"],
                       "gold_evidence": list(r.expert_sentences), "pred_evidence": res["evidence"],
                       "hallucinated_ids": res["hallucinated_ids"], "parse": res["parse"],
                       "raw_output": res["raw_output"], "explanation": res["explanation"],
                       "truncated_sentences": res["truncated_sentences"], "timing": res["timing"]}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
            done = len(rows)
            print(f"{done}/{len(df)} rows  acc so far {sum(x['gold_label']==x['pred_label'] for x in rows)/done:.3f}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    rep = full_report(rows)
    rep["meta"] = {"model": args.model, "adapter": adapter, "split": args.split, "n": len(rows),
                   "decoding": f"greedy, max_new_tokens={args.max_new_tokens}, fp32, sdpa, KV cache + patient-prefix reuse",
                   "parse_status_counts": {s: sum(r['parse'] == s for r in rows) for s in sorted({r['parse'] for r in rows})},
                   "hallucinated_id_records": sum(bool(r["hallucinated_ids"]) for r in rows),
                   "truncated_records": sum(bool(r["truncated_sentences"]) for r in rows),
                   "minutes": round((time.time() - t0) / 60, 2)}
    json.dump(rep, open(out_dir / f"metrics_{args.tag}_{args.split}.json", "w"), indent=2)
    o = rep["overall"]
    print(json.dumps({"accuracy": o["classification"]["accuracy"], "macro_f1": o["classification"]["macro_f1"],
                      "evidence_micro_f1": o["evidence"]["micro"]["f1"], **rep["meta"]}, indent=2))


if __name__ == "__main__":
    main()
