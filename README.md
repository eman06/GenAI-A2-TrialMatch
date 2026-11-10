# TrialMatch-Lite — criterion-level clinical-trial screening with a fine-tuned small Transformer

Generative AI — Assignment 2 (Eman Ihsan). An end-to-end, **CPU-only** system that reads a patient note and a
trial's inclusion/exclusion criteria and returns, for every criterion, one of the expert label categories
(`included / not included / excluded / not excluded / not enough information / not applicable`), the
**original sentence ids** that support the decision, the quoted evidence and a grounded explanation.
It also implements and validates the required inference mechanisms itself:

| Requirement | Where |
|---|---|
| Own Llama/SmolLM2 decoder (RoPE, GQA, RMSNorm, tied head) that loads the official weights | `trialmatch/model.py` |
| LoRA (own implementation) + shared-prefix training with activation checkpointing | `trialmatch/model.py`, `scripts/train_lora.py` |
| Fixed 37/8/8 patient split (seed 42) + manifest + leakage checks | `scripts/make_split.py`, `splits/split_manifest.json` |
| KV-cached decoding (pre-allocated + concat caches), uncached reference | `trialmatch/generation.py` |
| Reference attention and tiled online-softmax attention | `trialmatch/attention.py` |
| Greedy speculative decoding with cache roll-back | `trialmatch/generation.py::speculative_greedy` |
| Label strategies: generated JSON / constrained label scoring / calibrated scoring | `trialmatch/pipeline.py`, `scripts/calibrate.py` |
| Metrics (accuracy, macro-F1, per-class, confusion, evidence micro/macro) | `trialmatch/metrics.py`, `scripts/evaluate.py` |
| Browser application (screening, inference lab, live experiment view, reports) | `app/`, `start_app.bat` |
| Benchmarks | `scripts/bench_kv.py`, `scripts/bench_attention.py`, `scripts/bench_spec.py` → `results/*.json` |
| IEEE LaTeX report | `report/main.tex` (+ `report_overleaf.zip` built by `scripts/make_overleaf_zip.py`) |

> **Educational screening prototype — not a clinical enrollment decision tool.**

## Key results (held-out test split, 8 patients, 206 criteria)

| Model / label strategy | Accuracy | Macro-F1 | Evidence micro-F1 |
|---|---|---|---|
| SmolLM2-360M base, generated JSON | 0.413 | 0.200 | 0.000 (5.8% unparseable) |
| + LoRA fine-tuning, generated JSON | 0.626 | 0.285 | 0.021 (100% valid JSON) |
| + LoRA, calibrated label scoring (biases fit on validation only) | 0.437 | **0.344** | 0.021 |

* KV cache: identical tokens to the uncached reference in every setting; 3.0x-12.6x faster (64-512 input tokens).
* Tiled online-softmax attention: max error 1.3e-6 vs float64; with 256-wide tiles faster than the explicit reference from S=512.
* Speculative decoding (135M draft): token-identical in all 18 runs; 1.24-1.38x faster on 64-token generation,
  slower (0.76-0.79x for k=1 and k=4) on the 14-token JSON answers - explained in the report.
* 42 pytest tests pass (`results/pytest_summary.txt`).
* Training note: the planned 2-epoch run was interrupted at step 115 on the 8 GB laptop; the validated epoch-1
  checkpoint (validation loss 0.975 -> 0.185) is the released model.

---

## 1. Setup (Windows / Linux / macOS, no GPU needed)

```bash
git clone https://github.com/eman06/GenAI-A2-TrialMatch.git
cd GenAI-A2-TrialMatch
python -m pip install -r requirements.txt        # Python 3.10-3.11, torch 2.1 CPU is enough
python scripts/download_models.py                # base models (~1 GB) + dataset + LoRA adapter
```

Models are stored outside the repo in `~/.cache/trialmatch_models` (override with the `TRIALMATCH_MODELS`
environment variable). The LoRA adapter (`adapter.pt`, 35 MB) is published as a GitHub release asset and is
downloaded into `checkpoints/`; you can also re-train it (Section 3). The draft adapter is optional; without
it the base 135M draft is used (as in all reported results).

Tested hardware: Intel Core i5-1335U laptop (10 cores, 12 threads), 7.7 GB RAM, no GPU, Windows 11,
Python 3.11.9, torch 2.1.2, transformers 4.35.2 (only used for the tokenizer and for cross-checks).

## 2. Run the application (no IDE / notebook needed)

* **Windows:** double-click `start_app.bat`. A console window starts the server and the browser opens
  `http://127.0.0.1:8000` automatically.
* **Linux/macOS:** `./start_app.sh`

The header shows `model: loading...` for ~20 s, then `ready`. Tabs:

1. **Screen a patient** - paste/load a note, type criteria (one per line), fetch them from
   ClinicalTrials.gov by NCT id, or paste a whole eligibility section; pick a decoding mode; run.
   Each criterion shows the label (calibrated label scoring; the generated JSON label is shown when it
   differs), the cited sentences (highlighted in the note), an explanation and flags (no evidence,
   hallucinated ids, truncation). The preliminary patient-trial outcome is a *rule-based aggregation*
   (R1 blocking criterion -> likely ineligible; R2 missing information -> undetermined; R3 otherwise ->
   potentially eligible). Download the report as HTML or JSON.
2. **Inference lab** - runs one criterion through uncached / KV-cached / tiled-attention /
   speculative decoding and shows output equality, prefill/decode time, tok/s and acceptance;
   a live micro-benchmark of the three attention kernels with numerical error.
3. **Experiment results** - training curves, before/after metrics, KV/speculative/attention
   benchmarks, rendered live from the log files in `results/`.
4. **Saved reports** - every screening run is stored in `results/app_reports/` with provenance.

Invalid input (empty note, no criteria, malformed NCT id, network failure, model still loading) is
reported in the page without crashing; see `app/server.py`.

## 3. Reproduce the experiments

```bash
python scripts/make_split.py                                  # split + manifest + leakage checks
python scripts/train_lora.py --model SmolLM2-360M-Instruct --epochs 2 --chunk 6
bash   scripts/run_all_eval.sh                                # base/fine-tuned on val, then test (--final)
python scripts/calibrate.py                                   # label-scoring biases fit on validation only
python scripts/error_analysis.py finetuned test
python scripts/prompt_stats.py                                # token lengths + truncation audit
python scripts/bench_attention.py
python scripts/bench_kv.py --repeats 2
python scripts/bench_spec.py --n_prompts 3 --ks 1,2,4 --repeats 2
python scripts/make_diagrams.py && python scripts/make_figures.py   # report/figures + report/tables
python scripts/check_latex.py && python scripts/make_overleaf_zip.py
python -m pytest tests -q                                     # 42 correctness tests (~11 min on CPU)
```

All randomness is seeded (split: `numpy.default_rng(42)`; training: seed 42). The test split is
only touched by commands carrying `--final`. Optional: fine-tune the draft with
`python scripts/train_lora.py --model SmolLM2-135M-Instruct --epochs 1 --chunk 8 --out checkpoints/lora_smollm2_135m_draft`.

## 4. Repository layout

```
trialmatch/   config, data+prompts, model (decoder, caches, LoRA), attention kernels,
              generation (no-cache / KV / speculative), pipeline (parse, score, explain, aggregate), metrics
scripts/      split, training, evaluation, calibration, error analysis, benchmarks, figures, downloads
app/          FastAPI server + static HTML/JS front-end
tests/        pytest suite (attention, decoder vs HF, cache equivalence, speculative equality, split)
splits/       split_manifest.json (patient ids, counts, label counts, annotation ids)
results/      training logs, eval predictions+metrics, calibration, benchmark logs, error analysis
report/       IEEE LaTeX source, figures, tables
```

## 5. Data and leakage policy

Only `ncbi/TrialGPT-Criterion-Annotations` (1,015 annotations, 53 patients) is used for training and
evaluation. The GPT-4 columns (`gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`,
`explanation_correctness`) are dropped when the parquet is loaded and are never used as input or as a
training target (`tests/test_data_split.py`). One annotation has an empty criterion text; it is kept
(split integrity) with the placeholder "(criterion text missing)". No prompt was truncated (max 551 tokens).

## 6. AI-assistance statement

An AI coding assistant (Claude) was used to help write code, tests and report text. Every component
was executed and tested locally; the numerical results in `results/` come from these runs.

## 7. Troubleshooting

* `model failed to load` - run `python scripts/download_models.py`; check `TRIALMATCH_MODELS`.
* Out of memory on 8 GB machines - close other programs; training uses ~2.1 GB RSS with activation checkpointing.
* ClinicalTrials.gov fetch fails offline - paste criteria manually.
