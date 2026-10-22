"""LoRA SFT on GSM8K worked solutions.

    python scripts/sft.py --name sft-r16
    python scripts/sft.py --name smoke --max-steps 5 --limit 64     # quick check

Each example is a prompt ("Question: ...\\nAnswer:") and a completion (the
reference solution, ending "#### <answer>" and the EOS token). Loss is on the
completion only. The last 500 training problems are held out as a validation
set; the GSM8K test set is never touched here.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.gsm8k import PROMPT, clean_solution

VAL_SIZE = 500


def to_example(row):
    return {
        "prompt": PROMPT.format(question=row["question"]),
        "completion": " " + clean_solution(row["answer"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="first N training problems only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in checkpoints/<name>")
    ap.add_argument("--data", default=None,
                    help="JSONL of prompt/completion pairs to train on instead of the reference solutions "
                         "(e.g. from rft_sample.py); the validation set is unchanged")
    args = ap.parse_args()

    train = load_dataset("openai/gsm8k", "main", split="train")
    val = train.select(range(len(train) - VAL_SIZE, len(train)))
    train = train.select(range(len(train) - VAL_SIZE))
    if args.limit:
        train = train.select(range(args.limit))
        val = val.select(range(min(args.limit, VAL_SIZE)))
    train = train.map(to_example, remove_columns=train.column_names)
    val = val.map(to_example, remove_columns=val.column_names)
    if args.data:
        train = load_dataset("json", data_files=args.data, split="train")
        if args.limit:
            train = train.select(range(min(args.limit, len(train))))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        # Every linear layer in attention and the MLP, not just q/v.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    out_dir = Path("checkpoints") / args.name
    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float below 1 is a fraction of total steps
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=512,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    # Resuming restores the optimizer, scheduler, RNG state and data order, so
    # the run continues as if it had not stopped.
    trainer.train(resume_from_checkpoint=args.resume or None)

    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    metrics = trainer.evaluate()
    (out_dir / "final_eval.json").write_text(json.dumps({"args": vars(args), **metrics}, indent=2) + "\n")
    print(f"saved {final}; val loss {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
