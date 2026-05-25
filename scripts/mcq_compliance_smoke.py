#!/usr/bin/env python
"""MCQ-letter compliance smoke test for Qwen2.5-VL on VQA-RAD yes/no questions.

Decision-validating test of the LLaVA-Med-v1.0 -> Qwen2.5-VL pivot.

Background:
  The pivot was triggered partly by 0/11 MCQ-letter compliance on LLaVA-Med
  v1.0: when prompted with "Answer with the option's letter from the given
  choices directly", v1.0 never started a response with a letter, blocking
  use of standardized eval harnesses (VLMEvalKit, lmms-eval) that score
  by letter extraction. This script repeats that test against Qwen2.5-VL
  to validate the pivot before doing real evaluation work.

What it does:
  1. Loads the HF VQA-RAD test parquet
     (/data/dan/dataset/vqa_rad/data/test-*.parquet).
  2. Filters to yes/no questions (answer.lower() in {yes, no}). Strict
     subset of VQA-RAD's official closed split (excludes ~21 left/right
     questions); exact match to the A.Yes / B.No MCQ form.
  3. Samples N (default 20, seed-fixed).
  4. Loads Qwen2.5-VL-7B-Instruct (bf16, flash-attn 2, cuda:0).
  5. For each sample:
        - Decode JPEG bytes from the parquet row into a PIL Image.
        - Build a multimodal chat-template prompt with the
          HuatuoGPT-Vision MCQ instruction.
        - Greedy decode for up to --max-new-tokens.
        - Extract the first A/B via two regexes:
              strict  -- response starts with A or B
              lenient -- response contains an isolated A or B anywhere
  6. Reports strict %, lenient %, letter-correct %.
  7. Writes per-sample results + summary as JSON in results/ (unless
     --no-output).

Pass criterion:
  strict  >= 90%  -- pivot fully validated, no extractor needed
  lenient >= 90%  -- pivot validated; add letter-extractor in scoring
  else            -- investigate before proceeding
"""
import argparse
import io
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PARQUET = (
    "/data/dan/dataset/vqa_rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"
)
MCQ_TEMPLATE = (
    "{question}\n\n"
    "A. Yes\n"
    "B. No\n\n"
    "Answer with the option's letter from the given choices directly."
)


def gt_letter(answer: str):
    """yes -> A, no -> B, anything else -> None."""
    return {"yes": "A", "no": "B"}.get(answer.strip().lower())


def extract_letter(text: str):
    """Two-tier extraction.

    strict:  response starts with A or B, optionally followed by
             punctuation/whitespace (matches 'A', 'A.', 'A. Yes', 'B,').
    lenient: response contains an isolated A or B somewhere (matches
             'The answer is A.', 'Looking at the image, B').
    """
    if not text:
        return None, "empty"
    s = text.strip()
    if re.match(r"^([AB])([\s\.\),:]|$)", s):
        return s[0], "strict"
    m = re.search(r"\b([AB])\b", s)
    if m:
        return m.group(1), "lenient"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="Path to the VQA-RAD HuggingFace test parquet.")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16,
                    help="Generation cap. Letter-only responses need ~2-4 "
                         "tokens; 16 leaves headroom for verbose outputs "
                         "while still keeping the test fast.")
    ap.add_argument("--output-dir", default="results",
                    help="Where to write the JSON output. Relative paths "
                         "resolve from the current working directory.")
    ap.add_argument("--no-output", action="store_true",
                    help="Skip writing the JSON output file.")
    args = ap.parse_args()

    print("=" * 70)
    print("Qwen2.5-VL MCQ-letter compliance smoke test")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<16} {v}")
    print()

    # ---- Dataset ----------------------------------------------------------
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  Total test rows: {len(df)}")

    mask = df["answer"].str.strip().str.lower().isin(["yes", "no"])
    yesno = df[mask].reset_index(drop=True)
    print(f"  yes/no rows:     {len(yesno)}")

    random.seed(args.seed)
    sample_idx = random.sample(range(len(yesno)), min(args.n_samples, len(yesno)))
    samples = yesno.iloc[sample_idx].reset_index(drop=True)
    print(f"  Sampled:         {len(samples)} (seed={args.seed})")
    print()

    # ---- Model ------------------------------------------------------------
    print("Loading Qwen2.5-VL-7B-Instruct...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    ).eval()
    # use_fast=True selects the torchvision-v2 image processor (compiled
    # ops) over the default PIL-based slow path. Output equivalent up to
    # imperceptible floating-point noise; faster preprocessing.
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model_load_seconds = time.time() - t0
    print(f"  Loaded in {model_load_seconds:.1f}s\n")

    # Explicit generation config. Overriding temperature with None prevents
    # the "do_sample=False but temperature is set" warning that the model's
    # baked-in generation_config.json otherwise triggers.
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        temperature=None,
    )

    # ---- Inference loop ---------------------------------------------------
    results = []
    print(f"{'#':>3} {'gt':>3} {'pred':>5} {'mode':>8}  raw")
    print("-" * 70)
    t_inf_start = time.time()
    for i, row in samples.iterrows():
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        question = row["question"]
        gt = gt_letter(row["answer"])

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": MCQ_TEMPLATE.format(question=question)},
            ],
        }]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text], images=[image],
            padding=True, return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            out_ids = model.generate(**inputs, generation_config=gen_cfg)

        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        pred, mode = extract_letter(raw)
        results.append({
            "idx": int(i), "gt": gt, "pred": pred, "mode": mode,
            "raw": raw, "question": question,
            "answer": row["answer"],
        })
        print(f"{i:>3} {gt:>3} {str(pred):>5} {mode:>8}  {raw[:60]}")
    inference_seconds = time.time() - t_inf_start

    # ---- Summary ----------------------------------------------------------
    n = max(len(results), 1)
    n_strict  = sum(1 for r in results if r["mode"] == "strict")
    n_lenient = sum(1 for r in results if r["mode"] in ("strict", "lenient"))
    n_none    = sum(1 for r in results if r["mode"] == "none")
    n_correct = sum(1 for r in results
                    if r["pred"] is not None and r["pred"] == r["gt"])

    summary = {
        "scored": len(results),
        "strict_compliance":  {"count": n_strict,  "pct": round(100*n_strict/n,  1)},
        "lenient_compliance": {"count": n_lenient, "pct": round(100*n_lenient/n, 1)},
        "no_letter":          {"count": n_none,    "pct": round(100*n_none/n,    1)},
        "letter_correct":     {"count": n_correct, "pct": round(100*n_correct/n, 1)},
    }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Scored:                              {summary['scored']}")
    print(f"Strict  compliance  (^[AB][\\s.,:)]): "
          f"{summary['strict_compliance']['count']}/{n}  "
          f"({summary['strict_compliance']['pct']:.0f}%)")
    print(f"Lenient compliance  (\\b[AB]\\b):       "
          f"{summary['lenient_compliance']['count']}/{n}  "
          f"({summary['lenient_compliance']['pct']:.0f}%)")
    print(f"No letter found:                     "
          f"{summary['no_letter']['count']}/{n}  "
          f"({summary['no_letter']['pct']:.0f}%)")
    print(f"Letter-correct vs GT:                "
          f"{summary['letter_correct']['count']}/{n}  "
          f"({summary['letter_correct']['pct']:.0f}%)")
    print()
    if n_strict / n >= 0.9:
        verdict = "PASS (strict). Pivot validated."
    elif n_lenient / n >= 0.9:
        verdict = ("PASS (lenient). Pivot validated; "
                   "use a letter-extractor in scoring.")
    else:
        verdict = "FAIL. Investigate before proceeding."
    print(f"VERDICT: {verdict}")

    # ---- Save output ------------------------------------------------------
    if not args.no_output:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"mcq_compliance_smoke_{timestamp}.json"

        payload = {
            "script":   "scripts/mcq_compliance_smoke.py",
            "model_id": MODEL_ID,
            "config": {
                "parquet":        args.parquet,
                "n_samples":      args.n_samples,
                "seed":           args.seed,
                "max_new_tokens": args.max_new_tokens,
            },
            "timing": {
                "model_load_seconds": round(model_load_seconds, 2),
                "inference_seconds":  round(inference_seconds, 2),
                "ms_per_sample":      round(1000 * inference_seconds / n, 1),
            },
            "summary": summary,
            "verdict": verdict,
            "samples": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
