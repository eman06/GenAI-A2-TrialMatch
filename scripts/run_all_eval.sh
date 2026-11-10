#!/usr/bin/env bash
# Final evaluation chain (validation first, then the untouched test split once).
set -e
cd "$(dirname "$0")/.."
python -u scripts/evaluate.py --split val  --tag finetuned
python -u scripts/evaluate.py --split val  --tag base --adapter none
python -u scripts/evaluate.py --split test --tag finetuned --final
python -u scripts/evaluate.py --split test --tag base --adapter none --final
echo ALL_EVAL_DONE
