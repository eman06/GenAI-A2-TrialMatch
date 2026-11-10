"""B1/B2: fixed split, no leakage, GPT-4 columns never reach prompts."""
import json

import pandas as pd

from trialmatch.config import DATA_RAW, LEAKY_COLUMNS, SPLIT_MANIFEST
from trialmatch.data import load_annotations, make_split, parse_note, prefix_text, suffix_text, target_text


def test_split_reproducible_and_disjoint():
    df = load_annotations()
    sp = make_split(df)
    man = json.load(open(SPLIT_MANIFEST))
    for s in ("train", "val", "test"):
        assert sp[s] == man["splits"][s]["patient_ids"]
    assert [len(sp[s]) for s in ("train", "val", "test")] == [37, 8, 8]
    a, b, c = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (a & b) and not (a & c) and not (b & c)
    for s in ("train", "val", "test"):  # every annotation of a patient sits in exactly one split
        assert len(df[df.patient_id.isin(sp[s])]) == man["splits"][s]["n_annotations"]
    assert sum(man["splits"][s]["n_annotations"] for s in man["splits"]) == len(df) == 1015


def test_gpt4_columns_dropped_and_absent_from_prompt():
    df = load_annotations()
    assert not set(LEAKY_COLUMNS) & set(df.columns)
    raw = pd.read_parquet(DATA_RAW)
    for (_, r), (_, rr) in zip(df.head(50).iterrows(), raw.head(50).iterrows()):
        prompt = prefix_text(parse_note(r.note)) + suffix_text(r.criterion_type, r.criterion_text)
        assert rr.gpt4_explanation[:60] not in prompt
        assert "gpt" not in prompt.lower()


def test_sentence_ids_preserved():
    note = "0. First sentence.\n1. Second one.\n2. Third."
    assert parse_note(note) == [(0, "First sentence."), (1, "Second one."), (2, "Third.")]
    assert target_text("included", [0, 2]).startswith('{"label": "included", "evidence": [0, 2]}')
