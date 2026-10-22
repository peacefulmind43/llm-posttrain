"""Merge a LoRA adapter into its base model and save the result.

    python scripts/merge_lora.py --lora checkpoints/rft-r16/final --out checkpoints/rft-r16/merged

GRPO trains a fresh LoRA on top of the SFT model, so the SFT adapter is
merged first. The merged model is also the reference model for the KL term
(GRPO's LoRA is switched off to get reference log-probs).
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.lora).save_pretrained(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
