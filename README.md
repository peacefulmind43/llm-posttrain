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

## Success criteria (written 2026-10-22, before any training)

All numbers are strict accuracy on the full test set, 0-shot unless stated,
with Wilson 95% intervals.

- **SFT works** if the lower end of its interval is above the upper end of
  the 8-shot base model's interval (72.7%): with no examples in the prompt,
  it must beat the base model with eight.
- **GRPO works** if its interval does not overlap the SFT model's.
- **Guard rails**, for every model: at most 2% of answers without `####`,
  and mean output length at most twice the SFT model's (to catch reward
  hacking by rambling).

## Results

| model | strict | 95% CI | lenient | no `####` |
|---|---|---|---|---|
| base, 0-shot | 0.0% | [0.0, 0.3] | 13.4% | 100% |
| base, 8-shot | 70.3% | [67.8, 72.7] | 70.5% | 1.1% |

The base model never writes `####` without examples, and often runs on into
made-up questions; 0-shot strict accuracy measures format, not maths. The
8-shot score is the bar.

Status: work in progress.

## Run

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --limit 20 --backend hf --name smoke   # no GPU
python scripts/eval.py --model Qwen/Qwen3-1.7B-Base --shots 8 --name base-8shot            # GPU + vLLM
```
