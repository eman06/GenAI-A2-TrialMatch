"""Build every report figure and LaTeX table from the result logs (no hand-copied numbers)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import RESULTS, ROOT  # noqa: E402

FIG = ROOT / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "report" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2, C3, C4 = "#1f5fbf", "#c2410c", "#15803d", "#7c3aed"


def load(p):
    p = RESULTS / p
    return json.load(open(p)) if p.exists() else None


def tex(x):
    return str(x).replace("_", "\\_")


def f3(x):
    return f"{x:.3f}"


# ------------------------------------------------------------------ training curves
def training_curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for name, col in [("SmolLM2-360M-Instruct", C1), ("SmolLM2-135M-Instruct", C4)]:
        p = RESULTS / f"train_log_{name}.jsonl"
        if not p.exists():
            continue
        log = [json.loads(l) for l in open(p)]
        tr = np.array([(r["step"], r["loss"]) for r in log if "loss" in r])
        va = np.array([(r["step"], r["val_loss"]) for r in log if "val_loss" in r])
        k = 9
        sm = np.convolve(tr[:, 1], np.ones(k) / k, mode="valid")
        short = name.split("-")[1]
        ax.plot(tr[:, 0], tr[:, 1], color=col, alpha=.25, lw=.6)
        ax.plot(tr[k - 1:, 0], sm, color=col, lw=1.2, label=f"{short} train (avg 9)")
        ax.plot(va[:, 0], va[:, 1], "o--", color=col, ms=3, lw=1, label=f"{short} validation")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("answer-token CE loss"); ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "training_curves.pdf"); plt.close(fig)


# ------------------------------------------------------------------ evaluation
def eval_tables():
    runs = {}
    for p in sorted((RESULTS / "eval").glob("metrics_*.json")):
        runs[p.stem.removeprefix("metrics_")] = json.load(open(p))
    if not runs:
        return
    lines = []
    order = []
    for split in ("val", "test"):
        for model in ("base", "finetuned"):
            for strat, nm in (("", "generated"), ("-scored", "scored"), ("-calibrated", "calibrated")):
                order.append((f"{model}{strat}_{split}", f"{split} & {'base' if model == 'base' else 'LoRA'} & {nm}"))
    for key, name in order:
        if key not in runs:
            continue
        m = runs[key]
        o, i, e = m["overall"], m["inclusion"], m["exclusion"]
        if key.endswith("_test") and key.startswith("base_"):
            lines.append(r"\midrule")
        lines.append(" & ".join([name, f3(o["classification"]["accuracy"]),
                                 f3(o["classification"]["macro_f1"]), f3(i["classification"]["accuracy"]),
                                 f3(i["classification"]["macro_f1"]), f3(e["classification"]["accuracy"]),
                                 f3(e["classification"]["macro_f1"]), f3(o["evidence"]["micro"]["f1"]),
                                 f3(o["evidence"]["macro_nonempty_gold"]["f1"]),
                                 f3(o["classification"]["invalid_rate"])]) + r" \\")
    (TAB / "eval_main.tex").write_text("\n".join(lines))
    # per-class table for the fine-tuned test run (+ base for contrast)
    for key in ["finetuned-calibrated_test", "finetuned_test", "base_test", "finetuned_val"]:
        if key not in runs:
            continue
        rows = []
        for t in ("inclusion", "exclusion"):
            for lab, v in runs[key][t]["classification"]["per_class"].items():
                rows.append(f"{t[:4]}. & {lab} & {v['support']} & {f3(v['precision'])} & {f3(v['recall'])} & {f3(v['f1'])} \\\\")
        (TAB / f"per_class_{key}.tex").write_text("\n".join(rows))
        ev = runs[key]["overall"]["evidence"]
        evl = []
        for t in ("overall", "inclusion", "exclusion"):
            e = runs[key][t]["evidence"]
            evl.append(f"{t} & {f3(e['micro']['precision'])} & {f3(e['micro']['recall'])} & {f3(e['micro']['f1'])} & "
                       f"{f3(e['macro_nonempty_gold']['precision'])} & {f3(e['macro_nonempty_gold']['recall'])} & "
                       f"{f3(e['macro_nonempty_gold']['f1'])} & {e['macro_nonempty_gold']['n_records']} & "
                       f"{f3(e['empty_gold']['abstention_accuracy'] or 0)} \\\\")
        (TAB / f"evidence_{key}.tex").write_text("\n".join(evl))
    # confusion matrices
    for key in ["base_test", "finetuned_test", "finetuned-calibrated_test"]:
        if key not in runs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for ax, t in zip(axes, ("inclusion", "exclusion")):
            cm = runs[key][t]["classification"]["confusion"]
            M = np.array(cm["matrix"])
            ax.imshow(M, cmap="Blues")
            short = lambda s: {"not enough information": "NEI", "not applicable": "N/A", "not included": "not incl.",
                               "not excluded": "not excl."}.get(s, s)
            ax.set_xticks(range(len(cm["cols_pred"]))); ax.set_xticklabels([short(c) for c in cm["cols_pred"]], rotation=30, ha="right")
            ax.set_yticks(range(len(cm["rows_gold"]))); ax.set_yticklabels([short(c) for c in cm["rows_gold"]])
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    ax.text(b, a, M[a, b], ha="center", va="center", color="white" if M[a, b] > M.max() / 2 else "black", fontsize=7)
            ax.set_title(f"{t} ({key.replace('_test', '')})", fontsize=8)
            ax.set_xlabel("predicted"); ax.set_ylabel("expert")
        fig.tight_layout(); fig.savefig(FIG / f"confusion_{key}.pdf"); plt.close(fig)


# ------------------------------------------------------------------ KV cache
def kv_figs():
    kv = load("bench_kv.json")
    if not kv:
        return
    rows = []
    for r in kv["rows"]:
        nc = r.get("no_cache")
        rows.append(" & ".join([str(r["input_tokens"]), str(r["output_tokens"]),
                                f"{nc['total_s']['median']:.2f}" if nc else "--",
                                f"{r['kv_static']['total_s']['median']:.2f}", f"{r['kv_dynamic']['total_s']['median']:.2f}",
                                f"{r['kv_static']['prefill_s']:.3f}", f"{r['kv_static']['decode_s']:.2f}",
                                f"{r['kv_static']['decode_tokens_per_s']:.1f}", f"{nc['tokens_per_s']:.1f}" if nc else "--",
                                f"{r['kv_cache_analytic_mb']:.1f}",
                                f"{r['kv_static']['peak_rss_delta_mb']:.0f}/{nc['peak_rss_delta_mb']:.0f}" if nc else f"{r['kv_static']['peak_rss_delta_mb']:.0f}/--",
                                ("yes" if r.get("token_agreement") else "NO") if "token_agreement" in r else "n/a"]) + r" \\")
    (TAB / "kv.tex").write_text("\n".join(rows))
    ref = [r for r in kv["rows"] if "no_cache" in r]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(ref)); w = .27
    ax.bar(x - w, [r["no_cache"]["total_s"]["median"] for r in ref], w, color=C2, label="no cache")
    ax.bar(x, [r["kv_dynamic"]["total_s"]["median"] for r in ref], w, color=C4, label="KV (concat)")
    ax.bar(x + w, [r["kv_static"]["total_s"]["median"] for r in ref], w, color=C1, label="KV (pre-alloc.)")
    ax.set_xticks(x); ax.set_xticklabels([f"{r['input_tokens']}/{r['output_tokens']}" for r in ref])
    ax.set_xlabel("input / output tokens"); ax.set_ylabel("end-to-end latency (s)"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "kv_latency.pdf"); plt.close(fig)


# ------------------------------------------------------------------ speculative
def spec_figs():
    sp = load("bench_spec.json")
    if not sp:
        return
    lines = []
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for wl, mk in [("task", "o-"), ("long", "s--")]:
        rows = sp["workloads"].get(wl)
        if not rows:
            continue
        g_tps = np.mean([r["greedy"]["tokens_per_s"] for r in rows])
        keys = [k for k in rows[0] if k.startswith("draft_")]
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{wl} workload}} ({len(rows)} prompts, mean new tokens "
                     f"{np.mean([r['new_tokens'] for r in rows]):.0f}; greedy {g_tps:.1f} tok/s)}} \\\\")
        for d in ("draft_base_135m", "draft_lora_135m"):
            ks = sorted(int(k.split("_k")[1]) for k in keys if k.startswith(d))
            if not ks:
                continue
            for k in ks:
                xs = [r[f"{d}_k{k}"] for r in rows]
                acc = sum(x["accepted"] for x in xs) / max(sum(x["drafted"] for x in xs), 1)
                lines.append(" & ".join([tex(d.replace("draft_", "")), str(k),
                                         "yes" if all(x["equal_to_greedy"] for x in xs) else "NO", f3(acc),
                                         f"{np.mean([x['tokens_per_target_forward'] for x in xs]):.2f}",
                                         f"{np.mean([x['tokens_per_s'] for x in xs]):.1f}",
                                         f"{np.mean([x['speedup'] for x in xs]):.2f}"]) + r" \\")
            ax.plot(ks, [np.mean([r[f"{d}_k{k}"]["speedup"] for r in rows]) for k in ks], mk,
                    color=C1 if "base" in d else C3, ms=3, label=f"{wl}, {d.replace('draft_', '')}")
    ax.axhline(1, color="grey", lw=.8, ls=":")
    ax.set_xlabel("draft length k"); ax.set_ylabel("speed-up vs greedy KV"); ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "spec_speedup.pdf"); plt.close(fig)
    (TAB / "spec.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ attention
def attn_figs():
    at = load("bench_attention.json")
    if not at:
        return
    worst = {}
    for r in at["errors"]:
        w = worst.setdefault(r["impl"], [0, 0, 0])
        w[0] = max(w[0], r["max_abs_err"]); w[1] += r["mean_abs_err"]; w[2] += 1
    (TAB / "attn_err.tex").write_text("\n".join(
        f"{tex(k)} & {v[0]:.1e} & {v[1] / v[2]:.1e} \\\\" for k, v in worst.items()))
    rows = [r for r in at["speed"]]
    (TAB / "attn_speed.tex").write_text("\n".join(
        f"{r['T']} & {r['S']} & {tex(r['impl'])} & {1000 * r['t_median']:.2f} & {r['peak_rss_delta_mb']:.1f} & "
        f"{'fused' if r['analytic_intermediate_mb'] is None else format(r['analytic_intermediate_mb'], '.2f')} \\\\" for r in rows))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    sq = [r for r in rows if r["T"] == r["S"]]
    for impl, col in [("reference", C2), ("sdpa(fused)", C1), ("tiled_b64", C3), ("tiled_b256", C4)]:
        pts = [(r["S"], 1000 * r["t_median"]) for r in sq if r["impl"] == impl]
        if pts:
            ax.plot(*zip(*pts), "o-", color=col, ms=3, label=impl)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length (T = S)"); ax.set_ylabel("time (ms)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "attn_speed.pdf"); plt.close(fig)


def split_table_and_numbers():
    """tables/split.tex and numbers.tex (LaTeX macros for every number quoted in the text)."""
    man = json.load(open(ROOT / "splits" / "split_manifest.json"))
    from trialmatch.data import load_annotations
    df = load_annotations()
    rows = []
    for s in ("train", "val", "test"):
        sub = df[df.patient_id.isin(man["splits"][s]["patient_ids"])]
        rows.append(f"{s} & {sub.patient_id.nunique()} & {len(sub)} & {(sub.criterion_type == 'inclusion').sum()} & "
                    f"{(sub.criterion_type == 'exclusion').sum()} & {(sub.expert_sentences.str.len() == 0).sum()} \\\\")
    (TAB / "split.tex").write_text("\n".join(rows))
    N = {}
    ev = {p.stem.removeprefix("metrics_"): json.load(open(p)) for p in (RESULTS / "eval").glob("metrics_*.json")}
    for key, pre in [("base_test", "baseTest"), ("finetuned_test", "ftTest"), ("base_val", "baseVal"), ("finetuned_val", "ftVal"),
                     ("finetuned-calibrated_test", "calTest"), ("finetuned-calibrated_val", "calVal"),
                     ("base-scored_test", "baseScTest"), ("finetuned-scored_test", "ftScTest")]:
        if key in ev:
            o = ev[key]["overall"]
            N[pre + "Acc"] = f3(o["classification"]["accuracy"]); N[pre + "Fone"] = f3(o["classification"]["macro_f1"])
            N[pre + "EvF"] = f3(o["evidence"]["micro"]["f1"])
    kv = load("bench_kv.json")
    if kv:
        N["kvMaxSpeedup"] = f"{max(r.get('speedup_total', 0) for r in kv['rows']):.1f}"
    at = load("bench_attention.json")
    if at:
        N["tiledMaxErr"] = f"{max(r['max_abs_err'] for r in at['errors'] if r['impl'].startswith('tiled_fp32') and 'scaled' not in r['impl']):.1e}"
    stats = load("prompt_stats.json")
    if stats:
        N["promptMean"] = f"{stats['mean']:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items()]
    for k in ["calTestAcc", "calTestFone", "calTestEvF", "baseTestAcc", "ftTestAcc", "baseTestFone", "ftTestFone", "baseTestEvF", "ftTestEvF", "kvMaxSpeedup",
              "tiledMaxErr", "promptMean"]:
        if k not in N:
            lines.append(f"\\newcommand{{\\{k}}}{{TBD}}")
    (ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    training_curves(); eval_tables(); kv_figs(); spec_figs(); attn_figs(); split_table_and_numbers()
    print("figures:", sorted(p.name for p in FIG.glob("*")))
    print("tables:", sorted(p.name for p in TAB.glob("*")))
