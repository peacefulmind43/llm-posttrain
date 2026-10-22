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
| SFT (LoRA r16, 2 epochs), 0-shot | 65.7% | [63.1, 68.2] | 65.7% | 0.3% |

The base model never writes `####` without examples, and often runs on into
made-up questions; 0-shot strict accuracy measures format, not maths. The
8-shot score is the bar.

**SFT fails its bar.** It fixed the format (0.3% without `####`, against 100%
for the 0-shot base model) but it is 4.6 points *below* the 8-shot base model,
and the intervals do not overlap. On the same 1,319 problems the two models
are both right on 740, both wrong on 266; the base model alone is right on
187, the SFT model alone on 126. Validation loss flattened after about 400
steps (0.489 at step 100, 0.476 at 400, 0.477 at the end), so more epochs
would not help.

A likely reason: GSM8K's reference solutions are short and terse, written
for people, and the base model already reasons better in its own style.
Training on them by imitation pulls it towards that style and away from
some of what it could already do. GRPO starts from this SFT model, so it has
to recover that loss before it can show a gain.

Status: work in progress.

## Run

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --limit 20 --backend hf --name smoke   # no GPU
python scripts/eval.py --model Qwen/Qwen3-1.7B-Base --shots 8 --name base-8shot            # GPU + vLLM
```
