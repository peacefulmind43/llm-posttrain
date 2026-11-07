"""Keep only training problems the starting model sometimes gets right.

    python scripts/difficulty_filter.py --model checkpoints/rft-r16/merged --out data/mixed.jsonl

GRPO's advantage is the reward minus the group mean, so a problem where all
G answers are right (or all wrong) contributes nothing. In the first GRPO run
about half of all groups were like that. This script samples k answers per
training problem from the starting model (0-shot, temperature 1.0, as GRPO
does) and keeps the problems with between 1 and k-1 correct. Same split as
sft.py: the validation problems and the test set are never used.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.gsm8k import PROMPT, STOP, gold_answer, is_correct

VAL_SIZE = 500  # must match sft.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default="data/mixed.jsonl")
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    full = load_dataset("openai/gsm8k", "main", split="train")
    train = full.select(range(len(full) - VAL_SIZE))
    prompts = [PROMPT.format(question=r["question"]) for r in train]
    golds = [gold_answer(r["answer"]) for r in train]

    llm = LLM(model=args.model, seed=0, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, enable_prefix_caching=True)
    params = SamplingParams(n=args.k, temperature=args.temperature,
                            max_tokens=args.max_tokens, stop=STOP, seed=0)
    outs = llm.generate(prompts, params)

    counts = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p, gold, o in zip(prompts, golds, outs):
            c = sum(is_correct(x.text, gold) for x in o.outputs)
            counts.append(c)
            if 0 < c < args.k:
                f.write(json.dumps({"prompt": p, "answer": gold, "correct_of_k": c}) + "\n")

    hist = Counter(counts)
    kept = sum(v for c, v in hist.items() if 0 < c < args.k)
    stats = {
        "model": args.model, "k": args.k, "temperature": args.temperature,
        "problems": len(counts), "kept": kept,
        "all_correct": hist[args.k], "all_wrong": hist[0],
        "histogram": {str(c): hist[c] for c in range(args.k + 1)},
        "mean_accuracy": sum(counts) / (len(counts) * args.k),
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"kept {kept}/{len(counts)}; all correct {hist[args.k]}, all wrong {hist[0]}; "
          f"mean accuracy {stats['mean_accuracy']:.1%}")


if __name__ == "__main__":
    main()
