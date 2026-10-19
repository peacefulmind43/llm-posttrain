"""GSM8K loading, prompt format and answer checking.

The same checker is used for evaluation and, later, as the GRPO reward,
so a number the eval counts as correct is exactly a number the reward pays for.
"""

import re
from decimal import Decimal, InvalidOperation

# GSM8K solutions end with "#### <answer>". Models are taught (by few-shot
# examples, then SFT) to end the same way.
FINAL_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")
# Calculator annotations in the reference solutions, e.g. "<<48/2=24>>".
CALC_RE = re.compile(r"<<[^>]*>>")

PROMPT = "Question: {question}\nAnswer:"
# Stop generating when a base model starts writing the next few-shot question.
STOP = ["\n\nQuestion:", "\nQuestion:"]


def clean_solution(solution: str) -> str:
    """Reference solution without calculator annotations."""
    return CALC_RE.sub("", solution).strip()


def gold_answer(solution: str) -> str:
    m = FINAL_RE.search(solution)
    if m is None:
        raise ValueError("reference solution has no '####' answer")
    return normalize(m.group(1))


def normalize(number: str) -> str:
    """'1,000.00' -> '1000', '-3.50' -> '-3.5'. Raises ValueError if not a number."""
    try:
        d = Decimal(number.replace(",", ""))
    except InvalidOperation as e:
        raise ValueError(number) from e
    d = d.normalize()
    # Decimal('1E+3').normalize() prints in exponent form; force plain digits.
    return format(d, "f") if d != d.to_integral() else str(int(d))


def extract_answer(text: str, strict: bool = True) -> str | None:
    """The model's final answer, or None.

    strict: only an answer after '####' counts (used for the reward and the
    headline metric). Lenient: fall back to the last number in the text
    (reported alongside, to separate 'wrong maths' from 'wrong format').
    """
    m = FINAL_RE.search(text)
    if m is not None:
        raw = m.group(1)
    elif strict:
        return None
    else:
        nums = NUMBER_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    try:
        return normalize(raw)
    except ValueError:
        return None


def is_correct(text: str, gold: str, strict: bool = True) -> bool:
    return extract_answer(text, strict) == gold


def few_shot_prefix(train_rows, k: int) -> str:
    """The first k training examples as worked examples (fixed, so every run sees the same prompt)."""
    parts = []
    # Index row by row: slicing a Hugging Face Dataset returns columns, not rows.
    for row in (train_rows[i] for i in range(k)):
        parts.append(PROMPT.format(question=row["question"]) + " " + clean_solution(row["answer"]))
    return "\n\n".join(parts) + ("\n\n" if parts else "")


def build_prompt(question: str, prefix: str = "") -> str:
    return prefix + PROMPT.format(question=question)
