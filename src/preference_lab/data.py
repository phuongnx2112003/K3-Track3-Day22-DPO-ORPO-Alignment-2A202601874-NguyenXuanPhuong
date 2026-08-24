from __future__ import annotations

import json
import random
from pathlib import Path

from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load validated preference examples with line-numbered diagnostics."""
    examples: list[PreferenceExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        seen_prompts: dict[str, int] = {}
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                example = PreferenceExample.model_validate(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid preference example: {exc}") from exc
            key = " ".join(example.prompt.casefold().split())
            if key in seen_prompts:
                raise ValueError(f"{path}:{line_number}: duplicate prompt (first seen on line {seen_prompts[key]})")
            seen_prompts[key] = line_number
            examples.append(example)
    return examples

def save_jsonl(examples: list[PreferenceExample], path: str | Path) -> Path:
    """Write validated examples as canonical JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
    return target

def split_by_prompt(examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by normalized prompt to avoid leakage deterministically."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        key = " ".join(example.prompt.casefold().split())
        groups.setdefault(key, []).append(example)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_validation = max(1, round(len(keys) * validation_ratio)) if keys else 0
    validation_keys = set(keys[:n_validation])
    train = [item for key in keys if key not in validation_keys for item in groups[key]]
    validation = [item for key in keys if key in validation_keys for item in groups[key]]
    return train, validation
