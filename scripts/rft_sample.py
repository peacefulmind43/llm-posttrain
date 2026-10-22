"""Rejection sampling: build an SFT set from the base model's own correct solutions.

    python scripts/rft_sample.py --k 4 --out data/rft.jsonl

For each training problem (the same split as sft.py: the last 500 are held
out and never sampled, and the few-shot examples are skipped), draw k solutions from the base model with the 8-shot
prompt, keep one that the strict checker marks correct, and write it as a
prompt/completion pair in the same format sft.py uses. Problems with no
correct sample are dropped and counted.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.gsm8k import PROMPT, STOP, build_prompt, few_shot_prefix, gold_answer, is_correct

VAL_SIZE = 500  # must match sft.py
# Cut a sample right after its first "#### <number>" line: anything after it is
# the model starting the next question, and must not be trained on.
END_RE = re.compile(r"####\s*-?[\d,]*\.?\d+")


def trim(text: str) -> str:
    m = END_RE.search(text)
    return text[: m.end()] if m else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--k", type=int, default=4, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    full = load_dataset("openai/gsm8k", "main", split="train")
    # Skip the few-shot examples themselves: the model would just copy them.
    train = full.select(range(args.shots, len(full) - VAL_SIZE))
    if args.limit:
        train = train.select(range(args.limit))

    prefix = few_shot_prefix(full, args.shots)
    prompts = [build_prompt(r["question"], prefix) for r in train]
    golds = [gold_answer(r["answer"]) for r in train]

    llm = LLM(model=args.model, seed=0, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enable_prefix_caching=True)
    params = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                            max_tokens=512, stop=STOP, seed=0)
    outs = llm.generate(prompts, params)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    per_problem_correct = []
    with open(out, "w") as f:
        for row, gold, o in zip(train, golds, outs):
            texts = [trim(c.text) for c in o.outputs]
            correct = [t for t in texts if is_correct(t, gold)]
            per_problem_correct.append(len(correct))
            if correct:
                f.write(json.dumps({
                    "prompt": PROMPT.format(question=row["question"]),
                    "completion": " " + correct[0].strip(),
                }) + "\n")
                kept += 1

    n = len(train)
    total = n * args.k
    stats = {
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "shots": args.shots,
        "problems": n,
        "samples": total,
        "sample_accuracy": sum(per_problem_correct) / total,
        "problems_with_a_correct_sample": kept,
        "coverage": kept / n,
        "problems_all_k_correct": sum(c == args.k for c in per_problem_correct),
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"kept {kept}/{n} problems ({kept / n:.1%}); sample accuracy {stats['sample_accuracy']:.1%}")


if __name__ == "__main__":
    main()
