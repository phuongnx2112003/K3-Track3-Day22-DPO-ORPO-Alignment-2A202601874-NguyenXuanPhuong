from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import PreferenceExample


def pairwise_accuracy(examples: list[PreferenceExample], chosen_scores: list[float], rejected_scores: list[float]) -> float:
    """Return fraction where chosen score is greater than rejected score."""
    if not examples:
        return 0.0
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("score lengths must match the number of examples")
    wins = sum(1.0 if c > r else 0.5 if c == r else 0.0 for c, r in zip(chosen_scores, rejected_scores))
    return wins / len(examples)

def write_metrics(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_causal_model(model_dir: str | Path):
    """Load a causal LM for real model-derived evaluation."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the [train] dependencies to evaluate a model") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return torch, tokenizer, model


def sequence_logprob(torch, tokenizer, model, prompt: str, response: str, max_length: int) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(prompt + "\n" + response, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**encoded).logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logps[:, max(0, len(prompt_ids) - 1):].sum())


def model_pairwise_metrics(model_dir: str | Path, examples: list[PreferenceExample], max_length: int) -> dict[str, object]:
    torch, tokenizer, model = load_causal_model(model_dir)
    chosen = [sequence_logprob(torch, tokenizer, model, x.prompt, x.chosen, max_length) for x in examples]
    rejected = [sequence_logprob(torch, tokenizer, model, x.prompt, x.rejected, max_length) for x in examples]
    return {"pairwise_accuracy": pairwise_accuracy(examples, chosen, rejected), "examples": len(examples)}


def generate_response(torch, tokenizer, model, prompt: str, max_new_tokens: int = 64) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, min_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
