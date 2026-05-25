# ⛔ This repository is frozen

**Active development has moved to [huatuo-llava-v15-med-pruning](https://github.com/Leokuan0208/huatuo-llava-v15-med-pruning).**

This repo holds the work from the project's brief Qwen2.5-VL phase
(May 21–25, 2026). The Qwen2.5-VL-7B-Instruct MCQ-letter compliance
smoke test passed on May 24 (20/20 strict, see
`scripts/mcq_compliance_smoke.py`), validating that the Qwen2.5-VL model
family drives standardized MCQ-letter evaluation correctly — the open
question after LLaVA-Med v1.0 failed the same test 0/11 on May 20.

On May 25, after the literature survey for the full reproduction suite,
the project pivoted again to HuatuoGPT-Vision-7B (Chen et al. 2024,
arXiv:2406.19280, LLaVA-v1.5 architecture). The reasons in short:

- HuatuoGPT-Vision has published Table numbers on VQA-RAD, SLAKE,
  PathVQA, PMC-VQA, OmniMedVQA, and MMMU-Medical-Tracks. Qwen2.5-VL-7B-
  Instruct does not have published Table numbers on these benchmarks
  with a comparable methodology, so it cannot serve as a reproduction
  target on its own.
- HuatuoGPT-Vision's bundled `eval.py` + `Medical_Multimodal_Evaluation_Data`
  is a one-command reproduction pipeline. No new eval code to write.
- HuatuoGPT-Vision is a *medical* VLM, more on-thesis for a project
  titled "Question-Aware Visual Token Pruning for Medical VLMs."
- LLaVA-v1.5 is architecturally similar to LLaVA-Med v1.0, so the
  May 17 pruning hooks (`qsim`, `random`) from the original frozen
  `llava-med-pruning-v1` repo port with minor changes rather than a
  full rewrite.

The May 24 commits and the validated smoke-test artifact stay preserved
here as part of the methodology trail.

---
# medical-vlm-pruning

Research code for **Question-Aware Visual Token Pruning for Medical VLMs**.
Base model: Qwen2.5-VL-7B-Instruct. Evaluation harness: VLMEvalKit /
lmms-eval with MCQ-letter scoring on VQA-RAD, SLAKE, and PathVQA.

## Layout

- `scripts/` — entry points (smoke tests, baselines, sweeps)
- `pruning/` — pruning methods (`random`, `qsim`, future scorers)
- `eval/` — evaluation harness wrappers and the MCQ scorer

## Prior LLaVA-Med v1.0 work

The project's earlier LLaVA-Med v1.0 line of work (May 10–21, 2026) is
frozen at
**[Leokuan0208/llava-med-pruning-v1](https://github.com/Leokuan0208/llava-med-pruning-v1)**.
The pivot to Qwen2.5-VL was triggered by a substring-match bug in
v1.0's closed-set scorer (inflated accuracy) and 0/11 MCQ-letter
compliance on an instruction-following smoke test (blocked use of
standardized eval harnesses).

## Progress tracking

Day-by-day progress, experiment results, bug log, and related-work
notes:
[Leokuan0208/question-aware-vtp-medvlm](https://github.com/Leokuan0208/question-aware-vtp-medvlm).
Each day-page links the relevant commit in this repo.
