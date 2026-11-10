"""Timing and memory helpers for the benchmarks.

CPU has no allocator counter like ``torch.cuda.max_memory_allocated``, so on
CPU we sample the process resident set size (RSS) every millisecond from a
background thread and report the peak increase over the pre-run baseline.
This includes allocator caching effects and is therefore an upper bound on
the tensors actually needed; analytical sizes are reported next to it.
"""
from __future__ import annotations

import gc
import statistics
import threading
import time

import psutil
import torch


class PeakMemory:
    def __init__(self, device: str = "cpu", interval: float = 0.001):
        self.device, self.interval = device, interval
        self.peak_delta = 0

    def __enter__(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            self.base = torch.cuda.memory_allocated()
            return self
        self.proc = psutil.Process()
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.peak_delta = torch.cuda.max_memory_allocated() - self.base
        else:
            self._stop = True
            self._t.join()
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self.peak_delta = self.peak - self.base
        return False


def timed(fn, warmup: int = 1, repeats: int = 3, device: str = "cpu"):
    """Run fn() warmup times untimed, then `repeats` synchronised timed runs.
    Returns (list_of_seconds, last_result, peak_mem_bytes_of_last_run)."""
    for _ in range(warmup):
        fn()
    times, res, mem = [], None, 0
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        with PeakMemory(device) as pm:
            t0 = time.perf_counter()
            res = fn()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mem = max(mem, pm.peak_delta)
    return times, res, mem


def summarize(xs):
    return {"median": statistics.median(xs), "mean": statistics.fmean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}
