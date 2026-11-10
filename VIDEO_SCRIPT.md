# Demo video script (target 6 minutes)

Record with the Windows Game Bar (`Win + Alt + R`) or OBS, full screen browser + the console window.
Before recording: double-click `start_app.bat` once so the models are cached, close other apps.

---

### 0:00 – 0:35 Introduction
**Show:** the GitHub README, then double-click `start_app.bat`; the browser opens, the status badge turns green.

> "This is TrialMatch-Lite, my Assignment 2 system. It screens a patient note against each inclusion and exclusion
> criterion of a clinical trial, returns one of the expert label categories, and cites the patient sentences that
> support the decision. Everything runs locally on a laptop CPU with an open-weight model, SmolLM2-360M, that I
> fine-tuned with LoRA on the TrialGPT criterion annotations using the fixed 37/8/8 patient split.
> Note the banner: this is an educational screening prototype, not a clinical enrollment decision."

### 0:35 – 2:20 Screening workflow
**Do:** choose a validation example in "Load example", mode "Greedy + KV cache", click **Run screening**.

> "The note is split into numbered sentences; the IDs are kept so evidence can be checked against the expert
> annotations. The patient note is encoded once and its KV cache is reused for every criterion."

**Do:** click two criterion cards to highlight evidence; point at a *not enough information* card and a card flagged
"no supporting sentence".

> "Each card shows the predicted label, the raw JSON the model generated, the cited sentences and an explanation that
> only quotes the record. If the model makes a decision without citing anything, the app flags it for human
> verification instead of treating missing evidence as proof."

**Do:** point to the summary card.

> "The overall outcome is not a predicted ground truth – it is an explicit rule: any failed inclusion or met exclusion
> means likely ineligible, otherwise missing information means undetermined, otherwise potentially eligible."

**Do:** choose "Unseen hand-written case" (or type your own new note/criteria), run, click **Download report (HTML)** and open it.

> "This patient is not in the dataset. The report contains every decision, the evidence, the rule that was applied
> and provenance: model, adapter hash, decoding settings, time."

### 2:20 – 2:50 Input validation
**Do:** clear the note and click Run → error. Type `NCT123` and click Fetch → validation error. Restore the note.

> "Invalid input is reported clearly and nothing is lost; I can fix it and continue."

### 2:50 – 4:20 Inference lab
**Do:** tab 2, choose a criterion, **Compare decoding modes**.

> "The same prompt goes through four decoding paths I implemented: the uncached reference, which re-runs the whole
> sequence every step; my KV-cached decoder, which prefills once and then feeds one token per step with the correct
> RoPE position; the same with my tiled online-softmax attention; and greedy speculative decoding with the 135M draft
> model. All four produce identical text – the 'identical' column – but the cached path is several times faster.
> Speculative decoding shows the acceptance rate; on these very short JSON answers on a CPU it does not pay off,
> because the draft still costs a sizeable fraction of a target pass."

**Do:** attention micro-benchmark with S=T=1024, then S=1024, T=1.

> "Here the three attention kernels: explicit reference, my tiled version that keeps a running max and normaliser per
> row, and PyTorch's fused kernel. Errors are around 1e-6 against a float64 reference. The tiled Python loop is slower
> than the fused C++ kernel – its advantage is memory: it never builds the full score matrix."

### 4:20 – 5:40 Experiment results
**Do:** tab 3, scroll slowly.

> "These tables are read live from the log files. The training curve shows the validation loss falling from 0.97 to
> 0.19 after one epoch. Before fine-tuning the base model often fails to follow the schema; after fine-tuning every
> output is valid, but the model leans heavily on the majority labels, so I added constrained label scoring with
> biases calibrated on validation only, which improves macro-F1 on the untouched test set. Evidence localisation is
> the weakest part: the model mostly abstains from citing sentences.
> The KV benchmark confirms token-identical outputs and large speed-ups; the speculative benchmark reports
> acceptance, tokens per target pass and the measured slowdown or speed-up for each draft length."

### 5:40 – 6:15 Close
**Do:** tab 4 (saved reports).

> "Every run is stored with provenance. Limitations: a small model, a very small and imbalanced dataset, and CPU-only
> training. The code, split manifest, logs, adapter download and the IEEE report are in the GitHub repository. Thank you."
