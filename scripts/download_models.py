"""Download everything needed to run the app / reproduce results.

1. Base models (open weights, Apache-2.0) from the Hugging Face Hub:
     HuggingFaceTB/SmolLM2-360M-Instruct  (target)
     HuggingFaceTB/SmolLM2-135M-Instruct  (draft)
   into $TRIALMATCH_MODELS (default ~/.cache/trialmatch_models) - kept outside the repo.
2. The TrialGPT criterion-annotation parquet (if missing).
3. The trained LoRA adapters from the GitHub release (or skip with --no-adapters
   and train them yourself with scripts/train_lora.py).

    python scripts/download_models.py
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.config import (ADAPTER_DIR, DATA_RAW, DATASET_REPO, DRAFT_ADAPTER_DIR, HF_REPOS,  # noqa: E402
                               model_path)

# Release assets (fp32 LoRA state dicts, ~35 MB / ~19 MB). Edit if you fork the repo.
RELEASE = "https://github.com/eman06/GenAI-A2-TrialMatch/releases/download/v1.0"
ADAPTERS = {ADAPTER_DIR: f"{RELEASE}/adapter_smollm2_360m.pt",
            DRAFT_ADAPTER_DIR: f"{RELEASE}/adapter_smollm2_135m_draft.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-adapters", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    for name, repo in HF_REPOS.items():
        dst = model_path(name)
        if (dst / "model.safetensors").exists():
            print("ok  ", dst); continue
        print("get ", repo, "->", dst)
        snapshot_download(repo, local_dir=str(dst), local_dir_use_symlinks=False,
                          allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer*",
                                          "special_tokens_map.json", "vocab.json", "merges.txt"])
    if not Path(DATA_RAW).exists():
        Path(DATA_RAW).parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(DATASET_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
        Path(DATA_RAW).write_bytes(Path(p).read_bytes())
        print("dataset ->", DATA_RAW)
    if args.no_adapters:
        return
    for d, url in ADAPTERS.items():
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        if (d / "adapter.pt").exists():
            print("ok  ", d / "adapter.pt"); continue
        print("get ", url)
        try:
            urllib.request.urlretrieve(url, d / "adapter.pt")
        except Exception as e:  # noqa: BLE001
            print(f"  could not download adapter ({e}); train it with scripts/train_lora.py")


if __name__ == "__main__":
    main()
