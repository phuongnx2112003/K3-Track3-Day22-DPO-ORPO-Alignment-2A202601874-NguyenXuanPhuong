import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected

def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.25, seed=42)
    train_prompts = {" ".join(x.prompt.casefold().split()) for x in train}
    val_prompts = {" ".join(x.prompt.casefold().split()) for x in val}
    assert train_prompts.isdisjoint(val_prompts)


def test_load_rejects_duplicate_prompts(tmp_path) -> None:
    source = tmp_path / "duplicate.jsonl"
    source.write_text(
        '{"prompt":"Same prompt","chosen":"one","rejected":"two"}\n'
        '{"prompt":" same   prompt ","chosen":"three","rejected":"four"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate\.jsonl:2"):
        load_jsonl(source)


def test_load_reports_invalid_line_number(tmp_path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text("{\"prompt\":\"valid\",\"chosen\":\"yes\",\"rejected\":\"no\"}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid\.jsonl:2"):
        load_jsonl(source)
