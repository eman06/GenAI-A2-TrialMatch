"""Label-decision strategies compared on identical model outputs (A3 / C1).

  generated   : label parsed from the greedy JSON output
  scored      : argmax_l log P(label l | prompt)  (constrained to the allowed labels)
  calibrated  : argmax_l [log P(l | prompt) + b_type,l]

The per-type biases b are fitted ONLY on the validation predictions by
coordinate ascent on validation macro-F1, then applied unchanged to test.
Evidence ids always come from the generated JSON. Writes
results/calibration.json and metrics_<tag>-<strategy>_<split>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import LABELS, RESULTS  # noqa: E402
from trialmatch.metrics import classification_report, full_report  # noqa: E402

GRID = np.round(np.arange(-6, 6.01, 0.25), 2)


def load(tag, split):
    p = RESULTS / "eval" / f"preds_{tag}_{split}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def decide(rows, bias=None):
    out = []
    for r in rows:
        b = (bias or {}).get(r["criterion_type"], {})
        lp = r["label_logprobs"]
        out.append(max(lp, key=lambda l: lp[l] + b.get(l, 0.0)))
    return out


def fit(rows):
    bias = {}
    for t in ("inclusion", "exclusion"):
        sub = [r for r in rows if r["criterion_type"] == t]
        gold = [r["gold_label"] for r in sub]
        b = {l: 0.0 for l in LABELS[t]}
        score = lambda bb: classification_report(gold, decide(sub, {t: bb}), LABELS[t])["macro_f1"]
        best = score(b)
        for _ in range(3):                      # coordinate ascent passes
            for l in LABELS[t][1:]:             # first label is the reference (bias 0)
                for v in GRID:
                    trial = dict(b, **{l: float(v)})
                    s = score(trial)
                    if s > best + 1e-9:
                        best, b = s, trial
        bias[t] = b
        print(t, "val macro-F1 after calibration", round(best, 4), b)
    return bias


def write(tag, split, rows, labels, strategy):
    rr = [dict(r, pred_label=l) for r, l in zip(rows, labels)]
    rep = full_report(rr)
    rep["meta"] = {"split": split, "n": len(rr), "strategy": strategy, "source": f"preds_{tag}_{split}.jsonl",
                   "decoding": "greedy JSON for evidence; label from " + strategy}
    json.dump(rep, open(RESULTS / "eval" / f"metrics_{tag}-{strategy}_{split}.json", "w"), indent=2)
    o = rep["overall"]["classification"]
    print(f"{tag:10s} {split:4s} {strategy:10s} acc {o['accuracy']:.3f} macroF1 {o['macro_f1']:.3f}")


def main():
    cal = {}
    for tag in ("finetuned", "base"):
        val = load(tag, "val")
        if not val:
            continue
        bias = fit(val)
        cal[tag] = bias
        for split in ("val", "test"):
            rows = load(tag, split)
            if not rows:
                continue
            write(tag, split, rows, decide(rows), "scored")
            write(tag, split, rows, decide(rows, bias), "calibrated")
    out = {"method": "per-type additive log-prob biases fitted on validation macro-F1 (coordinate ascent, grid 0.25)",
           "bias": cal.get("finetuned"), "bias_base": cal.get("base")}
    json.dump(out, open(RESULTS / "calibration.json", "w"), indent=2)


if __name__ == "__main__":
    main()
