"""Common evaluation metrics (Task 2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import LABELS


def classification_report(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy, macro-F1 and per-class P/R/F1 + confusion matrix.

    Macro-F1 averages over the classes that have gold support in the evaluated
    subset (a class absent from gold has undefined recall). Predictions outside
    the label set (e.g. 'invalid' = unparseable output) count as errors and are
    shown as an extra confusion-matrix column.
    """
    gold, pred = list(gold), list(pred)
    labels = labels or sorted(set(gold))
    supported = [l for l in labels if l in set(gold)]
    acc = float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else 0.0
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    per_class = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, l in enumerate(labels)}
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in supported])) if supported else 0.0
    cols = labels + sorted(set(pred) - set(labels))
    cm = confusion_matrix(gold, pred, labels=cols)[: len(labels)]
    return {"n": len(gold), "accuracy": acc, "macro_f1": macro_f1, "macro_over": supported,
            "per_class": per_class, "confusion": {"rows_gold": labels, "cols_pred": cols, "matrix": cm.tolist()},
            "invalid_rate": float(np.mean([x not in labels for x in pred])) if pred else 0.0}


def evidence_scores(gold_sets: list[list[int]], pred_sets: list[list[int]]) -> dict:
    """Sentence-level evidence retrieval.

    micro : TP/FP/FN summed over ALL records. A record with empty gold evidence
            contributes only false positives (any cited id is unsupported).
    macro : mean of per-record P/R/F1 over records WITH gold evidence
            (empty prediction -> P=R=F1=0 for that record).
    empty-gold records are reported separately: 'abstention accuracy' = share
    where the model also cited nothing.
    """
    tp = fp = fn = 0
    per = []
    empty_total = empty_ok = 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if g:
            pr = len(g & p) / len(p) if p else 0.0
            rc = len(g & p) / len(g)
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            per.append((pr, rc, f1))
        else:
            empty_total += 1
            empty_ok += not p
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    per = np.array(per) if per else np.zeros((1, 3))
    return {"micro": {"precision": mp, "recall": mr, "f1": mf, "tp": tp, "fp": fp, "fn": fn},
            "macro_nonempty_gold": {"precision": float(per[:, 0].mean()), "recall": float(per[:, 1].mean()),
                                    "f1": float(per[:, 2].mean()), "n_records": int(len(per))},
            "empty_gold": {"n_records": empty_total, "abstention_accuracy": empty_ok / empty_total if empty_total else None}}


def full_report(rows: list[dict]) -> dict:
    """rows: dicts with criterion_type, gold_label, pred_label, gold_evidence, pred_evidence."""
    out = {}
    for name, sub in [("overall", rows),
                      ("inclusion", [r for r in rows if r["criterion_type"] == "inclusion"]),
                      ("exclusion", [r for r in rows if r["criterion_type"] == "exclusion"])]:
        if not sub:
            continue
        labels = (sorted(set(LABELS["inclusion"]) | set(LABELS["exclusion"])) if name == "overall" else LABELS[name])
        out[name] = {"classification": classification_report([r["gold_label"] for r in sub], [r["pred_label"] for r in sub], labels),
                     "evidence": evidence_scores([r["gold_evidence"] for r in sub], [r["pred_evidence"] for r in sub])}
    return out
