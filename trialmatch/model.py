"""A from-scratch Llama-architecture decoder (SmolLM2 family) with an explicit KV cache.

Why re-implement instead of calling ``AutoModelForCausalLM.generate``?
The assignment requires a cache-aware decoding flow that we control
(Tasks 3 and 5) and an attention kernel we can swap (Task 4). Owning the
forward pass makes positional handling (RoPE offsets), causal masking with a
cache, cache growth and cache roll-back explicit. ``tests/test_model.py``
checks that the logits match Hugging Face ``LlamaForCausalLM`` to ~1e-5.

The weights are the official SmolLM2 safetensors; LoRA adapters
(``LoRALinear``) are added for fine-tuning and can be merged for inference.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors.torch import load_file

from .attention import attention


# --------------------------------------------------------------------------- config
@dataclass
class LlamaConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "LlamaConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- KV caches
class StaticKVCache:
    """Pre-allocated KV cache: tensors of shape (L, B, H_kv, max_len, D).

    ``length`` = number of positions already written. Writing T new
    positions is an in-place slice assignment (O(T) per step, no
    re-allocation). Roll-back for speculative decoding is O(1): just move
    ``length`` back - stale entries are overwritten by the next write and are
    never read because attention only sees ``[:length]``.
    """

    def __init__(self, cfg: LlamaConfig, batch: int, max_len: int, dtype=torch.float32, device="cpu"):
        shape = (cfg.num_hidden_layers, batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        s, t = self.length, k_new.shape[2]
        if s + t > self.max_len:
            raise RuntimeError(f"KV cache overflow: {s}+{t} > {self.max_len}")
        self.k[layer, :, :, s:s + t] = k_new
        self.v[layer, :, :, s:s + t] = v_new
        return self.k[layer, :, :, :s + t], self.v[layer, :, :, :s + t]

    def advance(self, t: int):
        self.length += t

    def crop(self, length: int):
        assert 0 <= length <= self.length
        self.length = length

    def nbytes(self) -> int:
        """Bytes actually holding live keys/values."""
        per_pos = self.k[:, :, :, :1].numel() * self.k.element_size() * 2
        return per_pos * self.length

    def allocated_bytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


class DynamicKVCache:
    """Grow-by-concatenation cache (what a naive implementation does).

    Every step allocates new (…, S+T, D) tensors, so total copy cost over a
    generation of N tokens is O(N^2). Kept for comparison and because it
    keeps the autograd graph (used by the shared-prefix trainer).
    """

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.length = 0

    def update(self, layer, k_new, v_new):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]

    def advance(self, t):
        self.length += t

    def crop(self, length):
        self.k = [None if x is None else x[:, :, :length] for x in self.k]
        self.v = [None if x is None else x[:, :, :length] for x in self.v]
        self.length = length

    def expand_batch(self, b: int) -> "DynamicKVCache":
        """Share a batch-1 prefix cache across b sequences (views, autograd-safe)."""
        c = DynamicKVCache(len(self.k))
        c.k = [x.expand(b, -1, -1, -1) for x in self.k]
        c.v = [x.expand(b, -1, -1, -1) for x in self.v]
        c.length = self.length
        return c

    def nbytes(self):
        return sum(x.numel() * x.element_size() * 2 for x in self.k if x is not None)


class PrefixRecorder:
    """Training-time cache, phase 1 of shared-prefix batching: the batch-1 prefix
    forward stores each layer's K/V. ``update`` is idempotent and the object
    never changes mode, so re-running a layer during activation-checkpoint
    recomputation takes exactly the same code path."""

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.length = 0

    def update(self, layer, k_new, v_new):
        self.k[layer], self.v[layer] = k_new, v_new
        return k_new, v_new

    def advance(self, t):
        self.length = t


class PrefixReuser:
    """Phase 2: suffix tokens of B examples attend to [prefix ; own tokens]. The
    prefix K/V is broadcast with ``expand`` (no copy) and gradients flow back
    into the single prefix computation. Positions start at the prefix length."""

    def __init__(self, rec: PrefixRecorder):
        self.rec, self.length = rec, rec.length

    def update(self, layer, k_new, v_new):
        B = k_new.shape[0]
        return (torch.cat([self.rec.k[layer].expand(B, -1, -1, -1), k_new], dim=2),
                torch.cat([self.rec.v[layer].expand(B, -1, -1, -1), v_new], dim=2))

    def advance(self, t):
        pass


# --------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x  with frozen W, trainable A (r x in) and B (out x r)."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:
            y = y + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return y

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


# --------------------------------------------------------------------------- blocks
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with the 'rotate_half' (non-interleaved) layout used by Llama/SmolLM2."""

    def __init__(self, head_dim, theta):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions: torch.Tensor, dtype):
        # positions: (T,) absolute positions -> cos/sin of shape (T, D)
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, KV * D, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, KV * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache, backend):
        B, T, _ = x.shape
        H, KV, D = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)
        # RoPE is applied with the *absolute* positions of the new tokens
        # (past_len .. past_len+T-1) BEFORE caching, so cached keys are already
        # rotated and never need to be touched again.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)               # (B, KV, S, D)
        if KV != H:                                                  # grouped-query attention
            k = k.repeat_interleave(H // KV, dim=1)
            v = v.repeat_interleave(H // KV, dim=1)
        o = attention(q, k, v, backend)                              # causal w.r.t. q_start = S - T
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.self_attn = Attention(cfg, idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, h, cos, sin, cache, backend):
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, cache, backend)
        return h + self.mlp(self.post_attention_layernorm(h))


class LlamaDecoder(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.backend = "sdpa"

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids: torch.Tensor, cache=None, backend: str | None = None,
                logits_index: torch.Tensor | slice | None = None, start_pos: int | None = None,
                return_hidden: bool = False, checkpoint: bool = False):
        """input_ids (B, T). With a cache, the new tokens are placed at positions
        cache.length .. cache.length+T-1 and the cache is advanced by T.

        logits_index selects which time steps get projected to the vocabulary
        (e.g. ``slice(-1, None)`` during decoding) - the LM head is ~13% of the
        FLOPs of SmolLM2-360M, so skipping it for prompt tokens matters on CPU.
        """
        backend = backend or self.backend
        B, T = input_ids.shape
        past = cache.length if cache is not None else (start_pos or 0)
        pos = torch.arange(past, past + T, device=input_ids.device)
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(pos, h.dtype)
        for layer in self.layers:
            if checkpoint and torch.is_grad_enabled():
                # activation checkpointing: keep only layer inputs, recompute in backward
                h = torch.utils.checkpoint.checkpoint(layer, h, cos, sin, cache, backend, use_reentrant=False)
            else:
                h = layer(h, cos, sin, cache, backend)
        if cache is not None:
            cache.advance(T)
        if logits_index is not None:
            h = h[:, logits_index]
        h = self.norm(h)
        if return_hidden:
            return h
        return F.linear(h, self.embed_tokens.weight)                # tied LM head

    # ------------------------------------------------------------------ utilities
    def new_cache(self, max_len: int, batch: int = 1) -> StaticKVCache:
        p = next(self.parameters())
        return StaticKVCache(self.cfg, batch, max_len, p.dtype, p.device)

    def add_lora(self, r=16, alpha=32, dropout=0.05,
                 targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                 layers: range | None = None):
        for p in self.parameters():
            p.requires_grad_(False)
        layers = layers if layers is not None else range(self.cfg.num_hidden_layers)
        for i in layers:
            blk = self.layers[i]
            for parent in (blk.self_attn, blk.mlp):
                for name in targets:
                    if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, dropout))
        return self

    def lora_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if "lora_" in k}

    def merge_lora(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.merge()
        return self


# --------------------------------------------------------------------------- loading
def load_decoder(model_dir: str, dtype=torch.float32, device="cpu",
                 adapter_path: str | None = None, merge: bool = True) -> LlamaDecoder:
    cfg = LlamaConfig.from_json(os.path.join(model_dir, "config.json"))
    model = LlamaDecoder(cfg)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    sd = {k.removeprefix("model."): v.to(dtype) for k, v in sd.items() if k != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    if adapter_path:
        ad = torch.load(os.path.join(adapter_path, "adapter.pt"), map_location="cpu")
        acfg = ad["config"]
        model.add_lora(r=acfg["r"], alpha=acfg["alpha"], dropout=0.0, targets=tuple(acfg["targets"]),
                       layers=range(*acfg["layers"]))
        missing, unexpected = model.load_state_dict(ad["state_dict"], strict=False)
        if unexpected or any("lora_" in m for m in missing):
            raise RuntimeError("adapter does not match model")
        if merge:
            model.merge_lora()
    model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
