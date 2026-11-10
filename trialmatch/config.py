"""Project-wide constants and paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "trialgpt_criterion_annotations.parquet"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLIT_MANIFEST = ROOT / "splits" / "split_manifest.json"
RESULTS = ROOT / "results"
ADAPTER_DIR = ROOT / "checkpoints" / "lora_smollm2_360m"
DRAFT_ADAPTER_DIR = ROOT / "checkpoints" / "lora_smollm2_135m_draft"

# Models live outside the (OneDrive-synced) repo; override with TRIALMATCH_MODELS.
MODELS_DIR = Path(os.environ.get("TRIALMATCH_MODELS", Path.home() / ".cache" / "trialmatch_models"))
TARGET_MODEL = "SmolLM2-360M-Instruct"
DRAFT_MODEL = "SmolLM2-135M-Instruct"
HF_REPOS = {
    TARGET_MODEL: "HuggingFaceTB/SmolLM2-360M-Instruct",
    DRAFT_MODEL: "HuggingFaceTB/SmolLM2-135M-Instruct",
}

DATASET_REPO = "ncbi/TrialGPT-Criterion-Annotations"
SPLIT_SEED = 42
SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}

# Expert label sets are preserved per criterion type (no binarisation).
LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = sorted({l for v in LABELS.values() for l in v})

# Columns that hold GPT-4 outputs. They are dropped at load time and must never reach a prompt.
LEAKY_COLUMNS = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"]

MAX_PROMPT_TOKENS = 1024


def model_path(name: str) -> Path:
    return MODELS_DIR / name
