from src.gsm8k import build_prompt, clean_solution, extract_answer, few_shot_prefix, gold_answer, is_correct, normalize
from src.stats import wilson

SOLUTION = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nShe sold 48+24 = <<48+24=72>>72 clips.\n#### 72"


def test_gold_and_clean():
    assert gold_answer(SOLUTION) == "72"
    assert "<<" not in clean_solution(SOLUTION)
    assert clean_solution(SOLUTION).endswith("#### 72")


def test_normalize():
    assert normalize("1,000") == "1000"
    assert normalize("1000.00") == "1000"
    assert normalize("-3.50") == "-3.5"
    assert normalize("1E+3") == "1000"
    assert normalize(".5") == "0.5"


def test_strict_needs_marker():
    assert extract_answer("so the answer is 72") is None
    assert extract_answer("so the answer is 72", strict=False) == "72"
    assert extract_answer("... #### 72") == "72"
    assert extract_answer("#### $1,200") is None  # '$' after the marker is not a number
    assert extract_answer("####1,200.0") == "1200"


def test_first_marker_wins():
    # A model that keeps going after its answer is judged on the first one.
    assert extract_answer("#### 5\nQuestion: ... #### 9") == "5"


def test_is_correct():
    assert is_correct("work... #### 72", "72")
    assert not is_correct("work... #### 71", "72")
    assert not is_correct("the answer is 72", "72")
    assert is_correct("the answer is 72", "72", strict=False)


def test_prompt():
    rows = [{"question": "Q1?", "answer": SOLUTION}]
    prefix = few_shot_prefix(rows, 1)
    p = build_prompt("Q2?", prefix)
    assert p.startswith("Question: Q1?\nAnswer: Natalia")
    assert p.endswith("Question: Q2?\nAnswer:")
    assert build_prompt("Q2?") == "Question: Q2?\nAnswer:"


def test_wilson():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 10)[0] == 0.0 and wilson(0, 10)[1] > 0
    assert wilson(10, 10)[1] == 1.0 and wilson(10, 10)[0] < 1
