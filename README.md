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
