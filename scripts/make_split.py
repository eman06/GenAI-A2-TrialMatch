"""Create and verify the prescribed 37/8/8 patient-grouped split (seed 42).

Writes splits/split_manifest.json with patient ids, row counts and label
counts per split, and asserts there is no patient leakage.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import DATA_RAW, SPLIT_MANIFEST, SPLIT_SEED, LEAKY_COLUMNS
from trialmatch.data import load_annotations, make_split, parse_note

df = load_annotations()
for c in LEAKY_COLUMNS:
    assert c not in df.columns, c
splits = make_split(df, SPLIT_SEED)
sets = {k: set(v) for k, v in splits.items()}
assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
assert sum(len(v) for v in sets.values()) == df.patient_id.nunique() == 53
assigned = df.patient_id.map({p: s for s, ps in splits.items() for p in ps})
assert assigned.notna().all()
# every note sentence numbering is contiguous and evidence ids exist in the note
bad = 0
for _, r in df.iterrows():
    ids = [i for i, _ in parse_note(r.note)]
    assert ids == list(range(len(ids))), r.patient_id
    bad += any(e not in ids for e in r.expert_sentences)
man = {
    "dataset": "ncbi/TrialGPT-Criterion-Annotations",
    "raw_sha256": hashlib.sha256(Path(DATA_RAW).read_bytes()).hexdigest(),
    "procedure": "sorted unique patient_id -> numpy.random.default_rng(42).shuffle -> first 37 train, next 8 val, last 8 test",
    "dropped_columns": LEAKY_COLUMNS,
    "evidence_ids_outside_note": int(bad),
    "splits": {},
}
for s, pids in splits.items():
    sub = df[df.patient_id.isin(pids)]
    man["splits"][s] = {
        "n_patients": len(pids), "n_annotations": int(len(sub)), "n_trials": int(sub.trial_id.nunique()),
        "patient_ids": pids,
        "rows_per_patient": {p: int((sub.patient_id == p).sum()) for p in pids},
        "label_counts": {f"{t}/{l}": int(n) for (t, l), n in sub.groupby(["criterion_type", "expert_eligibility"]).size().items()},
        "annotation_ids": sorted(int(a) for a in sub.annotation_id),
    }
man["leakage_check"] = "passed: patient sets pairwise disjoint and cover all 53 patients"
Path(SPLIT_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
json.dump(man, open(SPLIT_MANIFEST, "w"), indent=2)
for s in splits:
    m = man["splits"][s]; print(s, m["n_patients"], "patients", m["n_annotations"], "rows", m["label_counts"])
print("evidence ids outside note:", bad)
