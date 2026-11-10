"""Categorise test-set errors of the fine-tuned model and pick representative cases (C4).

Categories (a record can fall into several):
  overclaim_missing_info : gold 'not enough information' but model gave a definite label
                           -> absent evidence treated as proof
  missed_evidence        : gold definite label but model said 'not enough information'
  polarity_flip          : met <-> not met (included<->not included, excluded<->not excluded)
  not_applicable_error   : gold or prediction is 'not applicable' and they differ
  unsupported_decision   : definite label with NO cited sentence
  contradictory_output   : 'not enough information' while citing evidence, or citing
                           evidence that is disjoint from the expert's evidence
  evidence_only_error    : label correct, evidence set differs from expert set
  hallucinated_ids / invalid_output
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import RESULTS  # noqa: E402
from trialmatch.data import load_split, parse_note  # noqa: E402

DEF = {"included", "not included", "excluded", "not excluded"}
FLIP = {frozenset({"included", "not included"}), frozenset({"excluded", "not excluded"})}


def categories(r):
    g, p = r["gold_label"], r["pred_label"]
    ge, pe = set(r["gold_evidence"]), set(r["pred_evidence"])
    c = []
    if p == "invalid":
        c.append("invalid_output")
    if g == "not enough information" and p in DEF:
        c.append("overclaim_missing_info")
    if g in DEF and p == "not enough information":
        c.append("missed_evidence")
    if frozenset({g, p}) in FLIP:
        c.append("polarity_flip")
    if g != p and "not applicable" in (g, p):
        c.append("not_applicable_error")
    if p in DEF and not pe:
        c.append("unsupported_decision")
    if (p == "not enough information" and pe) or (ge and pe and not (ge & pe)):
        c.append("contradictory_output")
    if g == p and ge != pe:
        c.append("evidence_only_error")
    if r["hallucinated_ids"]:
        c.append("hallucinated_ids")
    return c


def main(tag="finetuned", split="test"):
    rows = [json.loads(l) for l in open(RESULTS / "eval" / f"preds_{tag}_{split}.jsonl")]
    notes = {pid: dict(parse_note(sub.iloc[0].note)) for pid, sub in load_split()[split].groupby("patient_id")}
    cnt = Counter()
    picked, seen = [], Counter()
    for r in rows:
        cats = categories(r)
        r["categories"] = cats
        cnt.update(cats)
        if r["gold_label"] != r["pred_label"] or "evidence_only_error" in cats:
            main_cat = cats[0] if cats else "other_label_error"
            if seen[main_cat] < 2:
                seen[main_cat] += 1
                s = notes[r["patient_id"]]
                picked.append({"category": main_cat, "all_categories": cats, "annotation_id": r["annotation_id"],
                               "patient_id": r["patient_id"], "trial_id": r["trial_id"], "type": r["criterion_type"],
                               "criterion": r["criterion"][:300], "gold": r["gold_label"], "pred": r["pred_label"],
                               "gold_evidence": {i: s.get(i, "")[:200] for i in r["gold_evidence"]},
                               "pred_evidence": {i: s.get(i, "")[:200] for i in r["pred_evidence"]},
                               "raw_output": r["raw_output"]})
    n_err = sum(r["gold_label"] != r["pred_label"] for r in rows)
    out = {"tag": tag, "split": split, "n": len(rows), "n_label_errors": n_err, "category_counts": dict(cnt),
           "confusions": Counter(f"{r['gold_label']} -> {r['pred_label']}" for r in rows if r["gold_label"] != r["pred_label"]).most_common(),
           "examples": picked}
    json.dump(out, open(RESULTS / f"error_analysis_{tag}_{split}.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n", "n_label_errors", "category_counts")}, indent=1))
    print("examples:", len(picked))


if __name__ == "__main__":
    main(*sys.argv[1:])
