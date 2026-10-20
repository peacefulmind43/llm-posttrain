"""Evaluate a model on GSM8K test.

    python scripts/eval.py --model Qwen/Qwen3-1.7B-Base --shots 8 --name base-8shot
    python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --limit 20 --backend hf   # smoke test, no GPU

Greedy decoding, so a run is repeatable. Writes every generation to
results/<name>/generations.jsonl and the summary to results/<name>/summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.gsm8k import STOP, build_prompt, extract_answer, few_shot_prefix, gold_answer
from src.stats import wilson


def generate_vllm(model, prompts, max_tokens, lora, gpu_mem):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # 8-shot prompts are ~1.5k tokens; 4k leaves room for the answer and keeps
    # the KV cache small enough for an 8 GB card.
    llm = LLM(model=model, enable_lora=lora is not None, max_lora_rank=64, seed=0,
              gpu_memory_utilization=gpu_mem, max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=STOP)
    req = LoRARequest("adapter", 1, lora) if lora else None
    outs = llm.generate(prompts, params, lora_request=req)
    return [o.outputs[0].text for o in outs]


def generate_hf(model, prompts, max_tokens, lora, gpu_mem=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    if lora:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, lora)
    m.eval()
    texts = []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=max_tokens, do_sample=False, stop_strings=STOP, tokenizer=tok)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for s in STOP:
            text = text.split(s)[0]
        texts.append(text)
        print(f"  {i + 1}/{len(prompts)}", end="\r", flush=True)
    print()
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None, help="path to a LoRA adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N test questions only")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--gpu-mem", type=float, default=0.8, help="vLLM gpu_memory_utilization")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    train, test = ds["train"], ds["test"]
    if args.limit:
        test = test.select(range(args.limit))

    prefix = few_shot_prefix(train, args.shots)
    prompts = [build_prompt(r["question"], prefix) for r in test]
    golds = [gold_answer(r["answer"]) for r in test]

    gen = generate_vllm if args.backend == "vllm" else generate_hf
    texts = gen(args.model, prompts, args.max_tokens, args.lora, args.gpu_mem)

    out_dir = Path("results") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    strict = lenient = no_marker = 0
    with open(out_dir / "generations.jsonl", "w") as f:
        for r, gold, text in zip(test, golds, texts):
            s = extract_answer(text, strict=True)
            l = extract_answer(text, strict=False)
            strict += s == gold
            lenient += l == gold
            no_marker += s is None
            f.write(json.dumps({"question": r["question"], "gold": gold, "output": text, "strict": s, "lenient": l}) + "\n")

    summary = {
        "name": args.name,
        "model": args.model,
        "lora": args.lora,
        "shots": args.shots,
        "n": n,
        "strict_acc": strict / n,
        "strict_ci95": wilson(strict, n),
        "lenient_acc": lenient / n,
        "lenient_ci95": wilson(lenient, n),
        "no_marker_rate": no_marker / n,
        "mean_output_chars": sum(map(len, texts)) / n,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lo, hi = summary["strict_ci95"]
    print(f"{args.name}: strict {strict}/{n} = {strict / n:.1%} [{lo:.1%}, {hi:.1%}], "
          f"lenient {lenient / n:.1%}, no '####' {no_marker / n:.1%}")


if __name__ == "__main__":
    main()
