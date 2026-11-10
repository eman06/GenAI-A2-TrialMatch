# Submission checklist and demo script

## Deliverables (Google Classroom)

| Item | Location / status |
|---|---|
| Report PDF (IEEE, LaTeX) | `report/main.tex` → compile on Overleaf (upload `report_overleaf.zip`) → `report/main.pdf` |
| GitHub URL | `https://github.com/eman06/GenAI-A2-TrialMatch` (push this folder) |
| Reproducible instructions | `README.md` §1–3 |
| Model weights / adapter link | GitHub release `v1.0`: `adapter_smollm2_360m.pt`, `adapter_smollm2_135m_draft.pt` (`scripts/download_models.py` downloads them) |
| Split manifest | `splits/split_manifest.json` |
| Benchmark logs | `results/bench_kv.json`, `results/bench_attention.json`, `results/bench_spec.json`, `results/train_log_*.jsonl`, `results/eval/*` |
| Demo video (5–7 min) | record with the script below, upload (YouTube unlisted / Drive), paste link into `\video` in `report/main.tex` |
| AI-use appendix | report Appendix A |

## Before submitting

1. Create the GitHub repo `GenAI-A2-TrialMatch`, push, create release `v1.0` with the two adapter files from `checkpoints/*/adapter.pt` (renamed as above).
2. Put the video link in `report/main.tex` (`\newcommand{\video}`) and recompile.
3. Double-click `start_app.bat` once to make sure the app starts on the demo machine.

## 6-minute demo script

1. **0:00–0:30** Problem and disclaimer banner. Show `start_app.bat` starting the server and the browser opening; status turns *ready*.
2. **0:30–2:30 Screening.** Load the validation example, run with *KV cache*. Click criteria to highlight evidence sentences; point out labels including *not enough information*; the aggregated outcome and the rule (R1/R2/R3); the "no evidence – verify" flags. Then load the **unseen hand-written** example (or type a new note), run, download the HTML report and open it.
3. **2:30–3:00 Robustness.** Click Run with an empty note → error message; enter `NCT123` → validation error; recover by fixing input.
4. **3:00–4:30 Inference lab.** Compare decoding modes on one criterion: identical outputs, KV vs no-cache time, speculative acceptance stats. Run the attention micro-benchmark at S=1024 and explain why the tiled Python kernel is slower than fused SDPA but uses only a tile of memory.
5. **4:30–5:45 Experiment results tab.** Training/validation loss curve, before vs after fine-tuning table, KV benchmark (token agreement, speed-up), speculative table (acceptance, speed-up or slowdown and why).
6. **5:45–6:15** Saved reports tab with provenance; close with limitations.
