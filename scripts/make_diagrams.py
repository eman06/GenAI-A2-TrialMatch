"""Architecture / algorithm diagrams for the report (drawn with matplotlib -> vector PDF)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#1f5fbf", fs=5.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", )


def arrow(ax, a, b, txt=None, color="#333"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color))
    if txt:
        ax.text((a[0] + b[0]) / 2, max(a[1], b[1]) + 0.12, txt, ha="center", fontsize=5, color="#555")


def system():
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.1, 3.6, 2.3, 1.2, "Patient note\n(numbered sentences\n0..n-1)", "#fff7e6", "#c2410c")
    box(ax, 0.1, 1.9, 2.3, 1.2, "Trial criteria\n(type, text) list\n/ NCT id fetch", "#fff7e6", "#c2410c")
    box(ax, 3.0, 3.6, 2.5, 1.2, "Prefix prompt\nsystem + note\n(shared by all criteria)")
    box(ax, 3.0, 1.9, 2.5, 1.2, "Suffix prompt\ntype + criterion +\nallowed labels")
    box(ax, 6.1, 2.4, 3.0, 2.2, "SmolLM2-360M decoder\n(own implementation)\n32 x [RMSNorm, GQA attn\n+RoPE, SwiGLU MLP]\n+ merged LoRA (r=16)", "#e8f5ec", "#15803d", bold=False)
    box(ax, 6.1, 0.2, 3.0, 1.6, "Decoding paths\nno-cache | KV cache |\ntiled attention |\nspeculative (135M draft)", "#f3e8ff", "#7c3aed")
    box(ax, 9.8, 3.6, 2.0, 1.2, 'JSON parse\n{"label","evidence"}\n+ validation')
    box(ax, 9.8, 1.9, 2.0, 1.2, "Evidence by\noriginal ids +\ngrounded explanation")
    box(ax, 12.2, 2.6, 1.9, 1.6, "Rule-based\naggregation\nR1 / R2 / R3\n+ report export", "#fff7e6", "#c2410c")
    arrow(ax, (2.4, 4.2), (3.0, 4.2)); arrow(ax, (2.4, 2.5), (3.0, 2.5))
    arrow(ax, (5.5, 4.2), (6.1, 3.9), "KV once"); arrow(ax, (5.5, 2.5), (6.1, 2.9), "per criterion")
    arrow(ax, (7.6, 2.4), (7.6, 1.8)); arrow(ax, (9.1, 1.0), (10.8, 1.9))
    arrow(ax, (9.1, 3.9), (9.8, 4.2), "tokens"); arrow(ax, (10.8, 3.6), (10.8, 3.1))
    arrow(ax, (11.8, 2.5), (12.2, 3.1)); arrow(ax, (11.8, 4.2), (12.2, 3.7))
    fig.savefig(FIG / "system.pdf"); plt.close(fig)


def spec():
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.set_xlim(0, 7); ax.set_ylim(0, 5.4); ax.axis("off")
    toks = ["x", "d1", "d2", "d3", "d4"]
    for i, t in enumerate(toks):
        box(ax, 0.3 + i * 1.0, 4.2, 0.8, 0.6, t, "#f3e8ff", "#7c3aed")
    ax.text(5.5, 4.5, "draft: k=4\nsequential passes", fontsize=6, va="center")
    for i, t in enumerate(["g0", "g1", "g2", "g3", "g4"]):
        box(ax, 0.3 + i * 1.0, 2.7, 0.8, 0.6, t, "#e8f5ec", "#15803d")
    ax.text(5.5, 3.0, "target: ONE pass\nover [x,d1..d4]", fontsize=6, va="center")
    for i in range(5):
        arrow(ax, (0.7 + i, 4.2), (0.7 + i, 3.3))
    marks = ["d1=g0 ✓", "d2=g1 ✓", "d3≠g2 ✗", "", ""]
    for i, m in enumerate(marks[:3]):
        ax.text(1.7 + i - 1.0, 2.35, m, ha="center", fontsize=5.5, color="#15803d" if "✓" in m else "#b3261e")
    box(ax, 0.3, 1.0, 5.0, 0.7, "commit d1, d2, g2   (n = 2 accepted + 1 target token)", "#fff7e6", "#c2410c")
    box(ax, 0.3, 0.05, 5.0, 0.75, "roll back: target cache -> len(seq)-1,\ndraft cache -> accepted prefix", "#eef3fb", "#1f5fbf", fs=5.6)
    arrow(ax, (2.1, 2.1), (2.1, 1.7))
    fig.savefig(FIG / "spec_diagram.pdf"); plt.close(fig)


def tiled():
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.set_xlim(0, 7); ax.set_ylim(0, 5); ax.axis("off")
    n = 6
    for i in range(n):
        for j in range(n):
            if j > i:
                fc = "#f2f2f2"
            elif j == i:
                fc = "#fde7c8"
            else:
                fc = "#dbe7fb"
            ax.add_patch(plt.Rectangle((0.3 + j * 0.6, 4.3 - i * 0.6), 0.58, 0.58, fc=fc, ec="#888", lw=.4))
    ax.text(2.1, 4.95, "key tiles  ->", ha="center", fontsize=6)
    ax.text(0.05, 2.6, "query tiles", rotation=90, va="center", fontsize=6)
    ax.text(4.2, 4.4, "blue: full tile (no mask)", fontsize=5.5)
    ax.text(4.2, 4.0, "orange: diagonal (mask)", fontsize=5.5)
    ax.text(4.2, 3.6, "grey: skipped (future)", fontsize=5.5)
    ax.text(4.2, 2.9, "per query row keep:\n m = running max\n l = sum exp(s - m)\n acc = sum exp(s - m) v\nrescale by exp(m_old - m_new)\nout = acc / l", fontsize=5.5, va="top")
    fig.savefig(FIG / "tiled_diagram.pdf"); plt.close(fig)


if __name__ == "__main__":
    system(); spec(); tiled()
    print(sorted(p.name for p in FIG.glob("*diagram*")) + ["system.pdf"])
