"""Task 4: numerical validation + speed/memory of reference vs tiled vs fused attention.

Correctness is measured against a float64 reference on several shapes,
including prefill (T == S), chunked prefill with a cache (T < S) and
single-token decode (T == 1), all with the causal rule j <= (S - T) + i.

    python scripts/bench_attention.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.attention import reference_attention, sdpa_attention, tiled_attention  # noqa: E402
from trialmatch.bench_utils import PeakMemory, summarize  # noqa: E402
from trialmatch.config import RESULTS  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(10)
H, D = 15, 64      # SmolLM2-360M attention geometry

CASES = [  # (B, T, S) - small controlled cases first
    (1, 1, 1), (1, 7, 7), (2, 33, 33), (1, 5, 40), (1, 1, 129), (2, 64, 64), (1, 100, 100),
    (1, 128, 300), (1, 257, 257), (1, 512, 512), (1, 1024, 1024),
]
BLOCKS = [(16, 16), (64, 64), (128, 64)]


def errors():
    rows = []
    for B, T, S in CASES:
        q = torch.randn(B, H, T, D); k = torch.randn(B, H, S, D); v = torch.randn(B, H, S, D)
        gold = reference_attention(q.double(), k.double(), v.double())
        outs = {"reference_fp32": reference_attention(q, k, v), "sdpa_fp32": sdpa_attention(q, k, v)}
        for bq, bk in BLOCKS:
            outs[f"tiled_fp32_b{bq}x{bk}"] = tiled_attention(q, k, v, block_q=bq, block_k=bk)
        outs["tiled_fp64_b64x64"] = tiled_attention(q.double(), k.double(), v.double())
        # large-magnitude logits stress the online softmax (exp overflow without max subtraction)
        qs = q * 30
        gold_s = reference_attention(qs.double(), k.double(), v.double())
        outs_s = {"tiled_fp32_b64x64_scaled30": tiled_attention(qs, k, v), "reference_fp32_scaled30": reference_attention(qs, k, v)}
        for name, o in list(outs.items()) + list(outs_s.items()):
            g = gold_s if "scaled30" in name else gold
            err = (o.double() - g).abs()
            rows.append({"B": B, "T": T, "S": S, "impl": name, "max_abs_err": err.max().item(),
                         "mean_abs_err": err.mean().item(), "allclose_1e-5": bool(torch.allclose(o.double(), g, atol=1e-5, rtol=1e-4)),
                         "finite": bool(torch.isfinite(o).all())})
    return rows


def speed(repeats=5):
    rows = []
    for B, T, S in [(1, 128, 128), (1, 512, 512), (1, 1024, 1024), (1, 2048, 2048), (1, 1, 1024), (1, 1, 4096), (1, 64, 2048)]:
        q = torch.randn(B, H, T, D); k = torch.randn(B, H, S, D); v = torch.randn(B, H, S, D)
        impls = {"reference": lambda: reference_attention(q, k, v), "sdpa(fused)": lambda: sdpa_attention(q, k, v)}
        for bq, bk in [(64, 64), (128, 128), (256, 256)]:
            impls[f"tiled_b{bq}"] = (lambda bq=bq, bk=bk: tiled_attention(q, k, v, block_q=bq, block_k=bk))
        for name, fn in impls.items():
            fn()  # warm-up
            ts, mem = [], 0
            for _ in range(repeats):
                with PeakMemory() as pm:
                    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
                mem = max(mem, pm.peak_delta)
            # analytic size of the largest intermediate (fp32): full score+prob matrices vs one tile
            if name == "reference":
                inter = 2 * B * H * T * S * 4
            elif name.startswith("tiled"):
                bq = int(name.split("_b")[1]); inter = 2 * B * H * min(bq, T) * min(bq, S) * 4
            else:
                inter = None
            rows.append({"B": B, "T": T, "S": S, "impl": name, **{f"t_{k}": v for k, v in summarize(ts).items()},
                         "peak_rss_delta_mb": mem / 2**20, "analytic_intermediate_mb": None if inter is None else inter / 2**20})
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    err = errors()
    worst = {}
    for r in err:
        worst[r["impl"]] = max(worst.get(r["impl"], 0), r["max_abs_err"])
    print("worst max-abs error vs fp64 reference:", json.dumps(worst, indent=1))
    assert all(r["finite"] for r in err)
    sp = speed()
    RESULTS.mkdir(exist_ok=True)
    json.dump({"geometry": {"H": H, "D": D}, "errors": err, "speed": sp, "torch": torch.__version__,
               "threads": torch.get_num_threads()}, open(RESULTS / "bench_attention.json", "w"), indent=2)
