# Small-model post-training on GSM8K: SFT, then GRPO

## The question

How much does each post-training stage add to a small base model on maths
word problems: supervised fine-tuning (SFT) on worked solutions, then
reinforcement learning with GRPO and a reward that only checks the final answer?

## Setup

- **Model:** `Qwen/Qwen3-1.7B-Base` (pre-trained only, no chat tuning).
- **Data:** GSM8K, 7,473 training and 1,319 test problems. Every answer is a single number.
- **Stages:** base → LoRA SFT → GRPO, using Hugging Face TRL and PEFT.
- **Metric:** strict accuracy on the full test set. The answer must come
  after `####`, and the same checker is the GRPO reward. Greedy decoding,
  reported with Wilson 95% intervals. Lenient accuracy (last number in the
  output) is also reported, to separate format errors from maths errors.

Status: work in progress.

## Run

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --limit 20 --backend hf --name smoke   # no GPU
python scripts/eval.py --model Qwen/Qwen3-1.7B-Base --shots 8 --name base-8shot            # GPU + vLLM
```
