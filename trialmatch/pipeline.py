"""Criterion-level screening pipeline (Task 2) shared by evaluation and the web app.

prompt -> decoder (cached / uncached / speculative) -> JSON parse -> validation
-> evidence extraction by ORIGINAL sentence id -> grounded explanation
-> rule-based patient-trial aggregation.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import (ADAPTER_DIR, DRAFT_ADAPTER_DIR, DRAFT_MODEL, LABELS, MAX_PROMPT_TOKENS,
                     TARGET_MODEL, model_path)
from .data import encode_example, split_free_text
from .generation import greedy_kv, greedy_no_cache, speculative_greedy
from .model import load_decoder

IM_END = "<|im_end|>"
MET = {"inclusion": "included", "exclusion": "excluded"}
NOT_MET = {"inclusion": "not included", "exclusion": "not excluded"}


# --------------------------------------------------------------------------- parsing
def parse_output(text: str, ctype: str, valid_ids: set[int]) -> dict:
    """Robustly read {"label": ..., "evidence": [...]} from generated text.

    Returns label (or "invalid"), evidence ids that exist in the note, ids the
    model cited that do NOT exist (hallucinated), and a parse status.
    """
    raw = text.split(IM_END)[0].strip()
    label, evidence, status = None, [], "json"
    try:
        obj = json.loads(raw)
        label = str(obj.get("label", "")).strip().lower()
        evidence = obj.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
    except Exception:
        status = "regex"
        m = re.search(r'"label"\s*:\s*"([^"]*)"', raw)
        label = m.group(1).strip().lower() if m else raw.lower()
        m = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]?', raw)
        evidence = re.findall(r"\d+", m.group(1)) if m else []
    allowed = LABELS[ctype]
    if label not in allowed:  # tolerate extra words, longest label first ("not included" before "included")
        hit = next((l for l in sorted(allowed, key=len, reverse=True) if l in (label or "")), None)
        label, status = (hit, status + "+fuzzy") if hit else ("invalid", "invalid")
    ids = []
    for e in evidence:
        try:
            ids.append(int(e))
        except (TypeError, ValueError):
            pass
    ids = sorted(set(ids))
    return {"label": label, "evidence": [i for i in ids if i in valid_ids],
            "hallucinated_ids": [i for i in ids if i not in valid_ids], "parse": status, "raw": raw}


def explain(ctype: str, label: str, ev: list[tuple[int, str]]) -> str:
    """Template explanation that only restates the model decision and the cited
    patient sentences - it cannot introduce facts absent from the record."""
    quote = " ".join(f'[S{i}] "{s}"' for i, s in ev)
    if label == "invalid":
        return "The model output could not be parsed into an allowed label; treat this criterion as unresolved."
    if label == "not applicable":
        return ("The criterion was judged not applicable to this patient"
                + (f" based on {quote}." if ev else " (no specific sentence cited)."))
    if label == "not enough information":
        return ("The record does not contain enough information to decide this criterion"
                + (f"; related but inconclusive sentences: {quote}." if ev else
                   "; no sentence addresses it. Absence of a finding is not treated as proof either way."))
    met = label == MET[ctype]
    effect = {("inclusion", True): "supports inclusion", ("inclusion", False): "argues against inclusion",
              ("exclusion", True): "would exclude the patient", ("exclusion", False): "does not exclude the patient"}[(ctype, met)]
    if ev:
        return f"Judged '{label}': the cited record {quote} {effect}."
    return (f"Judged '{label}' ({effect}) but NO supporting sentence was cited - this rests on the absence of "
            "a finding and should be verified by a human reviewer.")


# --------------------------------------------------------------------------- aggregation
def aggregate(results: list[dict]) -> dict:
    """Explicit, rule-based patient-trial summary (NOT a ground-truth label).

    R1  any inclusion 'not included' or exclusion 'excluded'          -> LIKELY INELIGIBLE
    R2  else any 'not enough information' / 'invalid'                  -> UNDETERMINED (needs info)
    R3  else (all inclusion included|n/a, all exclusion not excluded|n/a) -> POTENTIALLY ELIGIBLE
    Unsupported decisions (no cited evidence) are listed for human verification.
    """
    blocking = [r for r in results if r["label"] in ("not included", "excluded")]
    missing = [r for r in results if r["label"] in ("not enough information", "invalid")]
    unsupported = [r for r in results if not r["evidence"] and r["label"] in ("included", "not included", "excluded", "not excluded")]
    if blocking:
        verdict, rule = "Likely ineligible (preliminary)", "R1: at least one inclusion criterion not met or exclusion criterion met"
    elif missing:
        verdict, rule = "Undetermined - more information needed", "R2: no blocking criterion, but some criteria lack information"
    else:
        verdict, rule = "Potentially eligible (preliminary)", "R3: all inclusion met/not applicable and no exclusion met"
    counts = {}
    for r in results:
        counts[f"{r['criterion_type']}/{r['label']}"] = counts.get(f"{r['criterion_type']}/{r['label']}", 0) + 1
    return {"verdict": verdict, "rule": rule, "counts": counts,
            "blocking": [r["criterion"] for r in blocking], "missing_information": [r["criterion"] for r in missing],
            "needs_verification": [r["criterion"] for r in unsupported]}


# --------------------------------------------------------------------------- screener
@dataclass
class ScreenerConfig:
    target: str = TARGET_MODEL
    adapter: str | None = str(ADAPTER_DIR)
    draft: str | None = DRAFT_MODEL
    draft_adapter: str | None = None
    max_new_tokens: int = 48
    backend: str = "sdpa"
    threads: int = 10
    label_strategy: str = "calibrated"     # "generated" or "calibrated" (needs results/calibration.json)


class Screener:
    def __init__(self, cfg: ScreenerConfig = ScreenerConfig()):
        torch.set_num_threads(cfg.threads)
        self.cfg = cfg
        tdir = str(model_path(cfg.target))
        self.tok = AutoTokenizer.from_pretrained(tdir)
        adapter = cfg.adapter if cfg.adapter and Path(cfg.adapter, "adapter.pt").exists() else None
        self.adapter_used = adapter
        self.target = load_decoder(tdir, adapter_path=adapter)
        self.draft = None
        if cfg.draft:
            dadp = cfg.draft_adapter if cfg.draft_adapter and Path(cfg.draft_adapter, "adapter.pt").exists() else None
            self.draft = load_decoder(str(model_path(cfg.draft)), adapter_path=dadp)
            self.draft_adapter_used = dadp
        self.eos = self.tok.convert_tokens_to_ids(IM_END)
        cal = Path(__file__).resolve().parents[1] / "results" / "calibration.json"
        self.calibration = json.load(open(cal)) if cal.exists() and adapter else None

    # one criterion ---------------------------------------------------------
    def _generate(self, ids, mode, prefix_cache=None, k=4):
        if mode == "kv":
            return greedy_kv(self.target, ids, self.cfg.max_new_tokens, self.eos, self.cfg.backend, cache=prefix_cache)
        if mode == "nocache":
            return greedy_no_cache(self.target, ids, self.cfg.max_new_tokens, self.eos, self.cfg.backend)
        if mode == "speculative":
            if self.draft is None:
                raise ValueError("speculative mode needs a draft model")
            return speculative_greedy(self.target, self.draft, ids, self.cfg.max_new_tokens, k, self.eos, self.cfg.backend)
        if mode == "tiled":
            return greedy_kv(self.target, ids, self.cfg.max_new_tokens, self.eos, "tiled", cache=prefix_cache)
        raise ValueError(f"unknown mode {mode}")

    def screen_criterion(self, sents, ctype, ctext, mode="kv", prefix_cache=None, k=4) -> dict:
        enc = encode_example(self.tok, sents, ctype, ctext, max_prompt_tokens=MAX_PROMPT_TOKENS)
        ids = enc.prefix_ids + enc.suffix_ids
        if prefix_cache is not None:
            if getattr(prefix_cache, "prefix_ids", None) == enc.prefix_ids:
                prefix_cache.crop(len(enc.prefix_ids))  # reuse the patient prefix K/V, drop the last criterion
            else:                                       # truncation changed the prefix -> no reuse
                prefix_cache = None
        g = self._generate(ids, mode, prefix_cache, k)
        text = self.tok.decode(g.tokens)
        valid = {i for i, _ in sents if i not in set(enc.truncated_sentences)}
        p = parse_output(text, ctype, valid)
        t_s = time.perf_counter()
        scores = self.score_labels(ids, ctype, prefix_cache if mode in ("kv", "tiled") else None)
        score_s = time.perf_counter() - t_s
        generated_label = p["label"]
        label, strategy = generated_label, "generated"
        if self.calibration and self.cfg.label_strategy == "calibrated":
            b = self.calibration["bias"][ctype]
            label = max(scores, key=lambda l: scores[l] + b.get(l, 0.0))
            strategy = "calibrated label scoring"
        sent_map = dict(sents)
        ev = [(i, sent_map[i]) for i in p["evidence"]]
        return {
            "criterion": ctext, "criterion_type": ctype, "label": label, "generated_label": generated_label,
            "label_logprobs": {l: round(v, 4) for l, v in scores.items()}, "label_strategy": strategy,
            "evidence": p["evidence"], "evidence_text": [{"id": i, "text": s} for i, s in ev],
            "hallucinated_ids": p["hallucinated_ids"], "explanation": explain(ctype, label, ev),
            "raw_output": p["raw"], "parse": p["parse"], "truncated_sentences": enc.truncated_sentences,
            "timing": {"mode": mode, "prompt_tokens": len(ids), "new_tokens": len(g.tokens),
                       "prefill_s": round(g.prefill_s, 4), "decode_s": round(g.decode_s, 4),
                       "total_s": round(g.total_s, 4), "tokens_per_s": round(g.tokens_per_s, 2),
                       "label_scoring_s": round(score_s, 4),
                       **{k2: v for k2, v in g.stats.items() if k2 != "accept_hist"}},
        }

    @torch.no_grad()
    def score_labels(self, ids, ctype, cache=None) -> dict:
        """Constrained scoring: log P(label tokens | prompt, '{"label": "') for every allowed label.

        Uses the KV cache: the prompt is (re)used once, the shared JSON head is fed once,
        and each candidate label only costs one short forward after cropping the cache back.
        Tokenisation matches training: the label is followed by '",' as in the target JSON."""
        head_txt = '{"label": "'
        head = self.tok(head_txt, add_special_tokens=False).input_ids
        if cache is None:
            cache = self.target.new_cache(len(ids) + len(head) + 16)
            self.target(torch.tensor([ids]), cache=cache, logits_index=slice(0, 0))
        else:
            cache.crop(len(ids))
        hl = torch.log_softmax(self.target(torch.tensor([head]), cache=cache, logits_index=slice(-1, None))[0, -1], -1)
        base = cache.length
        out = {}
        for lab in LABELS[ctype]:
            full = self.tok(head_txt + lab + '",', add_special_tokens=False).input_ids
            assert full[:len(head)] == head, "tokenisation boundary mismatch"
            lt = full[len(head):]
            cache.crop(base)
            lp = hl[lt[0]].item()
            if len(lt) > 1:
                lo = torch.log_softmax(self.target(torch.tensor([lt[:-1]]), cache=cache)[0], -1)
                lp += sum(lo[i, lt[i + 1]].item() for i in range(len(lt) - 1))
            out[lab] = lp
        cache.crop(len(ids))
        return out

    def prefill_patient(self, sents):
        """Run the patient-note prefix once; later criteria start from this cache."""
        enc = encode_example(self.tok, sents, "inclusion", "x", max_prompt_tokens=MAX_PROMPT_TOKENS)
        cache = self.target.new_cache(MAX_PROMPT_TOKENS + self.cfg.max_new_tokens + 8)
        with torch.no_grad():
            self.target(torch.tensor([enc.prefix_ids]), cache=cache, logits_index=slice(0, 0))
        cache.prefix_ids = enc.prefix_ids
        return cache

    # whole patient ---------------------------------------------------------
    def screen(self, note: str, criteria: list[tuple[str, str]], mode="kv", k=4, reuse_prefix=True) -> dict:
        sents = split_free_text(note)
        if not sents:
            raise ValueError("patient note is empty")
        t0 = time.perf_counter()
        cache = self.prefill_patient(sents) if (reuse_prefix and mode in ("kv", "tiled")) else None
        prefix_s = time.perf_counter() - t0
        results = [self.screen_criterion(sents, ct, tx, mode, cache, k) for ct, tx in criteria]
        return {"sentences": [{"id": i, "text": s} for i, s in sents], "results": results,
                "summary": aggregate(results), "mode": mode, "prefix_prefill_s": round(prefix_s, 4),
                "total_s": round(time.perf_counter() - t0, 3),
                "model": {"target": self.cfg.target, "adapter": self.adapter_used,
                          "draft": self.cfg.draft if mode == "speculative" else None,
                          "decoding": f"greedy, max_new_tokens={self.cfg.max_new_tokens}, fp32, attention={self.cfg.backend if mode != 'tiled' else 'tiled'}"}}
