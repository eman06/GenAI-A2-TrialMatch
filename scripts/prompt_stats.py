"""Token-length statistics of the prompts per split and truncation audit (Task 1 documentation)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from trialmatch.config import MAX_PROMPT_TOKENS, RESULTS, TARGET_MODEL, model_path  # noqa: E402
from trialmatch.data import encode_example, load_split, parse_note  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(model_path(TARGET_MODEL)))
out, allp = {}, []
for s, df in load_split().items():
    P, S, T, tr = [], [], [], 0
    for _, r in df.iterrows():
        e = encode_example(tok, parse_note(r.note), r.criterion_type, r.criterion_text, r.expert_eligibility,
                           r.expert_sentences, MAX_PROMPT_TOKENS)
        P.append(len(e.prefix_ids)); S.append(len(e.suffix_ids)); T.append(len(e.target_ids)); tr += bool(e.truncated_sentences)
    prompt = np.array(P) + np.array(S)
    allp += prompt.tolist()
    out[s] = {"prefix_mean": float(np.mean(P)), "prefix_max": int(max(P)), "suffix_mean": float(np.mean(S)),
              "suffix_max": int(max(S)), "target_mean": float(np.mean(T)), "target_max": int(max(T)),
              "prompt_mean": float(prompt.mean()), "prompt_max": int(prompt.max()), "truncated": tr}
out["mean"] = float(np.mean(allp)); out["max"] = int(max(allp)); out["limit"] = MAX_PROMPT_TOKENS
json.dump(out, open(RESULTS / "prompt_stats.json", "w"), indent=2)
print(json.dumps(out, indent=1))
