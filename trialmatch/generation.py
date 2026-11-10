"""Greedy decoding paths (Tasks 3 and 5).

* ``greedy_no_cache``   - reference: re-runs the full sequence every step (O(N^2) FLOPs).
* ``greedy_kv``         - prefill once, then feed ONE token per step against the KV cache.
* ``speculative_greedy``- draft k tokens with a small model, verify them in one target
                          forward, accept the longest matching prefix + 1 target token.

All three return a ``GenResult`` with token ids and timing so the benchmarks
and the web app can compare them on identical inputs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .model import DynamicKVCache, LlamaDecoder


@dataclass
class GenResult:
    tokens: list[int]
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return len(self.tokens) / self.total_s if self.total_s > 0 else 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        # first token comes out of the prefill pass; the rest from decode steps
        n = max(len(self.tokens) - 1, 0)
        return n / self.decode_s if self.decode_s > 0 else 0.0


def _sync(device):
    if isinstance(device, torch.device):
        device = device.type
    if device == "cuda":
        torch.cuda.synchronize()


def _argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits[0, -1]).item())


# --------------------------------------------------------------------------- reference
@torch.no_grad()
def greedy_no_cache(model: LlamaDecoder, prompt: list[int], max_new: int, eos: int | None = None,
                    backend: str | None = None) -> GenResult:
    """No cache reuse: step t recomputes attention over all prompt+t tokens."""
    dev = next(model.parameters()).device
    seq = list(prompt)
    out = []
    _sync(dev); t0 = time.perf_counter()
    prefill_s = 0.0
    for i in range(max_new):
        logits = model(torch.tensor([seq], device=dev), backend=backend, logits_index=slice(-1, None))
        nxt = _argmax(logits)
        if i == 0:
            _sync(dev); prefill_s = time.perf_counter() - t0
        out.append(nxt); seq.append(nxt)
        if eos is not None and nxt == eos:
            break
    _sync(dev); total = time.perf_counter() - t0
    return GenResult(out, prefill_s, total - prefill_s, total)


# --------------------------------------------------------------------------- KV cache
@torch.no_grad()
def greedy_kv(model: LlamaDecoder, prompt: list[int], max_new: int, eos: int | None = None,
              backend: str | None = None, cache_kind: str = "static", cache=None) -> GenResult:
    """Cache-aware greedy decoding.

    prefill : one forward over the whole prompt writes K/V for positions 0..P-1
              and yields the logits for the first new token.
    decode  : each step feeds only the newest token; it is given position
              cache.length (RoPE) and attends to every cached key - no mask is
              needed because a single query at the end may see all past keys.
    An optional pre-filled ``cache`` (e.g. a patient-note prefix) can be passed;
    then only ``prompt[cache.length:]`` is processed in the prefill.
    """
    dev = next(model.parameters()).device
    if cache is None:
        cache = (model.new_cache(len(prompt) + max_new + 1) if cache_kind == "static"
                 else DynamicKVCache(model.cfg.num_hidden_layers))
    _sync(dev); t0 = time.perf_counter()
    logits = model(torch.tensor([prompt[cache.length:]], device=dev), cache=cache, backend=backend,
                   logits_index=slice(-1, None))
    nxt = _argmax(logits)
    _sync(dev); t1 = time.perf_counter()
    out = [nxt]
    while len(out) < max_new and not (eos is not None and nxt == eos):
        logits = model(torch.tensor([[nxt]], device=dev), cache=cache, backend=backend)
        nxt = _argmax(logits)
        out.append(nxt)
    _sync(dev); t2 = time.perf_counter()
    kv_bytes = cache.nbytes()
    return GenResult(out, t1 - t0, t2 - t1, t2 - t0, {"kv_cache_bytes": kv_bytes, "cache_len": cache.length})


# --------------------------------------------------------------------------- speculative
@torch.no_grad()
def speculative_greedy(target: LlamaDecoder, draft: LlamaDecoder, prompt: list[int], max_new: int,
                       k: int = 4, eos: int | None = None, backend: str | None = None) -> GenResult:
    """Greedy speculative decoding (Leviathan et al. 2023 / Chen et al. 2023, greedy case).

    Invariant at the start of every round:
        committed sequence  seq = prompt + out
        target cache holds K/V for seq[:-1]   (the last committed token is "pending")
        draft  cache holds K/V for a prefix of seq[:-1] (it may lag by one token)

    Round:
      1. Draft: catch the draft cache up to seq[:-1], then greedily propose
         d1..dk, feeding each proposal back into the draft (k draft forwards).
      2. Verify: ONE target forward over [seq[-1], d1, ..., dk] (k+1 tokens).
         Row i of its logits is the target's greedy choice g_i after d1..di.
      3. Accept: n = length of the longest prefix with d_{i+1} == g_i. Append
         d1..dn and the target's own token g_n (the correction if n < k, a free
         "bonus" token if n == k). The output is therefore exactly the target's
         greedy output.
      4. Roll back: crop the target cache to len(seq_new) - 1 (drop K/V of
         rejected drafts); crop the draft cache to the accepted part.
    """
    dev = next(target.parameters()).device
    max_len = len(prompt) + max_new + k + 2
    tcache, dcache = target.new_cache(max_len), draft.new_cache(max_len)
    stats = {"rounds": 0, "drafted": 0, "accepted": 0, "accept_hist": [0] * (k + 1),
             "target_forwards": 0, "draft_forwards": 0}
    _sync(dev); t0 = time.perf_counter()

    # prefill both models on the prompt; the target produces the first token
    logits = target(torch.tensor([prompt], device=dev), cache=tcache, backend=backend, logits_index=slice(-1, None))
    draft(torch.tensor([prompt[:-1]], device=dev), cache=dcache, backend=backend, logits_index=slice(-1, None))
    stats["target_forwards"] += 1; stats["draft_forwards"] += 1
    out = [_argmax(logits)]
    _sync(dev); t_prefill = time.perf_counter() - t0
    seq = list(prompt) + out
    # target cache now covers the full prompt = seq[:-1]: invariant holds

    while len(out) < max_new and not (eos is not None and out[-1] == eos):
        kk = min(k, max_new - len(out))          # never draft past the budget
        # ---- 1. draft
        feed = seq[dcache.length:]               # catch-up tokens incl. the pending one
        proposals = []
        for _ in range(kk):
            dl = draft(torch.tensor([feed], device=dev), cache=dcache, backend=backend, logits_index=slice(-1, None))
            stats["draft_forwards"] += 1
            d = _argmax(dl)
            proposals.append(d)
            feed = [d]
            if eos is not None and d == eos:
                break
        # ---- 2. verify (one target pass over pending token + proposals)
        base = tcache.length                     # == len(seq) - 1
        vl = target(torch.tensor([[seq[-1]] + proposals], device=dev), cache=tcache, backend=backend)
        stats["target_forwards"] += 1
        greedy = torch.argmax(vl[0], dim=-1).tolist()   # len(proposals)+1 target choices
        # ---- 3. accept longest matching prefix, then one target token
        n = 0
        while n < len(proposals) and proposals[n] == greedy[n]:
            n += 1
        new_tokens = proposals[:n] + [greedy[n]]
        stats["rounds"] += 1; stats["drafted"] += len(proposals); stats["accepted"] += n
        stats["accept_hist"][n] += 1
        # stop at EOS / budget
        if eos is not None and eos in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(eos) + 1]
        new_tokens = new_tokens[:max_new - len(out)]
        out += new_tokens; seq += new_tokens
        # ---- 4. cache roll-back: keep K/V only for seq[:-1]
        tcache.crop(min(tcache.length, len(seq) - 1))
        assert tcache.length == len(seq) - 1 or len(out) >= max_new or out[-1] == eos
        dcache.crop(min(dcache.length, len(seq) - 1))
    _sync(dev); total = time.perf_counter() - t0
    stats["acceptance_rate"] = stats["accepted"] / max(stats["drafted"], 1)
    stats["tokens_per_target_forward"] = len(out) / max(stats["target_forwards"], 1)
    stats["k"] = k
    return GenResult(out, t_prefill, total - t_prefill, total, stats)
