"""Dataset loading, the prescribed patient-grouped split, and prompt construction."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (DATA_RAW, LABELS, LEAKY_COLUMNS, SPLIT_MANIFEST, SPLIT_SEED,
                     SPLIT_SIZES)

SYSTEM_PROMPT = (
    "You are a clinical trial screening assistant. Decide whether ONE eligibility criterion "
    "is met by the patient, using only the numbered patient sentences. Missing information "
    "is not evidence for or against eligibility. Answer with JSON only."
)


# --------------------------------------------------------------------------- loading
def load_annotations(path=DATA_RAW) -> pd.DataFrame:
    """Load the TrialGPT criterion annotations and DROP every GPT-4 generated column."""
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    df["expert_sentences"] = df["expert_sentences"].apply(lambda s: sorted(set(int(x) for x in ast.literal_eval(s))))
    # one annotation (sigir-201519, exclusion) has a NULL criterion text; keep the row (split
    # integrity) but make the absence explicit in the prompt.
    df["criterion_text"] = df["criterion_text"].fillna("").str.strip().replace("", "(criterion text missing)")
    df["criterion_type"] = df["criterion_type"].str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].str.strip().str.lower()
    return df


def make_split(df: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Sort unique patient IDs, shuffle with default_rng(42), assign 37/8/8."""
    pids = np.array(sorted(df["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_tr, n_va, n_te = SPLIT_SIZES["train"], SPLIT_SIZES["val"], SPLIT_SIZES["test"]
    assert len(pids) == n_tr + n_va + n_te, f"expected 53 patients, found {len(pids)}"
    return {
        "train": [str(p) for p in pids[:n_tr]],
        "val": [str(p) for p in pids[n_tr:n_tr + n_va]],
        "test": [str(p) for p in pids[n_tr + n_va:]],
    }


def load_split(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = load_annotations() if df is None else df
    with open(SPLIT_MANIFEST) as f:
        man = json.load(f)
    out = {s: df[df.patient_id.isin(man["splits"][s]["patient_ids"])].reset_index(drop=True)
           for s in ("train", "val", "test")}
    return out


# --------------------------------------------------------------------------- notes
_SENT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def parse_note(note: str) -> list[tuple[int, str]]:
    """The dataset notes are already one numbered sentence per line ("0. ...").
    We keep the ORIGINAL ids so evidence can be scored against expert ids."""
    sents = []
    for line in note.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SENT_RE.match(line)
        if m:
            sents.append((int(m.group(1)), m.group(2).strip()))
        elif sents:  # continuation line -> append to previous sentence
            sid, txt = sents[-1]
            sents[-1] = (sid, txt + " " + line)
        else:
            sents.append((0, line))
    return sents


def split_free_text(note: str) -> list[tuple[int, str]]:
    """For unseen notes typed into the app: number sentences ourselves if not numbered."""
    if all(_SENT_RE.match(l.strip()) for l in note.splitlines() if l.strip()):
        return parse_note(note)
    text = re.sub(r"\s+", " ", note).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [(i, p.strip()) for i, p in enumerate(parts) if p.strip()]


def format_note(sents: list[tuple[int, str]]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sents)


# --------------------------------------------------------------------------- prompts
def prefix_text(sents) -> str:
    """Patient part of the prompt - identical for every criterion of a patient,
    which lets training and the app reuse its KV cache across criteria."""
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            f"Patient note (numbered sentences):\n{format_note(sents)}\n")


def suffix_text(criterion_type: str, criterion_text: str) -> str:
    labels = " | ".join(LABELS[criterion_type])
    return (f"\nCriterion type: {criterion_type}\nCriterion: {criterion_text.strip()}\n"
            f"Allowed labels: {labels}\n"
            'Reply as {"label": <one allowed label>, "evidence": [<supporting sentence numbers>]}'
            "<|im_end|>\n<|im_start|>assistant\n")


def target_text(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": list(evidence)}) + "<|im_end|>"


@dataclass
class Encoded:
    prefix_ids: list[int]
    suffix_ids: list[int]
    target_ids: list[int] = field(default_factory=list)
    truncated_sentences: list[int] = field(default_factory=list)


def encode_example(tok, sents, ctype, ctext, label=None, evidence=None, max_prompt_tokens=1024) -> Encoded:
    """Tokenise prefix / suffix / target separately and concatenate (same at train and test time).

    Truncation policy: if the prompt exceeds ``max_prompt_tokens`` we drop whole
    sentences from the END of the note (never cut mid-sentence, ids preserved)
    and report which sentence ids were removed so evidence loss is auditable.
    """
    suffix_ids = tok(suffix_text(ctype, ctext), add_special_tokens=False).input_ids
    kept = list(sents)
    dropped = []
    while True:
        prefix_ids = tok(prefix_text(kept), add_special_tokens=False).input_ids
        if len(prefix_ids) + len(suffix_ids) <= max_prompt_tokens or len(kept) <= 1:
            break
        dropped.append(kept.pop()[0])
    tgt = tok(target_text(label, evidence), add_special_tokens=False).input_ids if label is not None else []
    return Encoded(prefix_ids, suffix_ids, tgt, sorted(dropped))
