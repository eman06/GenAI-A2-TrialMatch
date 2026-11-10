#!/usr/bin/env bash
# After evaluation: calibration, error analysis, prompt stats, benchmarks, figures.
cd "$(dirname "$0")/.."
while ! grep -q ALL_EVAL_DONE results/eval_stdout.txt; do
  if grep -q Traceback results/eval_stdout.txt; then echo EVAL_FAILED; exit 1; fi
  sleep 15
done
python -u scripts/calibrate.py && echo STEP calibrate
python -u scripts/error_analysis.py finetuned test && echo STEP errors
python -u scripts/prompt_stats.py > /dev/null && echo STEP prompt_stats
python -u scripts/bench_attention.py > results/bench_attention_stdout.txt 2>&1 && echo STEP attention
python -u scripts/bench_kv.py --repeats 2 > results/bench_kv_stdout.txt 2>&1 && echo STEP kv
python -u scripts/bench_spec.py --n_prompts 3 --ks 1,2,4 --repeats 2 > results/bench_spec_stdout.txt 2>&1 && echo STEP spec
python -u scripts/make_figures.py && echo STEP figures
echo POST_DONE
