from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import generate_response, load_causal_model, model_pairwise_metrics, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")

@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")

@app.command()
def evaluate(config: Path) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"].get("validation_data", cfg["paths"]["train_data"]))
    model_dir = Path(cfg["paths"]["output_dir"]) / "dpo_model"
    if not model_dir.exists():
        raise typer.BadParameter(f"trained model not found: {model_dir}; run pref-lab train first")
    metrics = model_pairwise_metrics(model_dir, examples, cfg["training"]["max_length"])
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote model-derived metrics to {out}: {metrics}[/green]")

@app.command()
def prepare(data: Path, output_dir: Path = Path("data/processed"), validation_ratio: float = 0.2, seed: int = 42) -> None:
    """Validate, canonicalize, and split data before fine-tuning."""
    from .data import save_jsonl, split_by_prompt
    examples = load_jsonl(data)
    train, validation = split_by_prompt(examples, validation_ratio, seed)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(validation, output_dir / "validation.jsonl")
    print(f"[green]Prepared {len(train)} train and {len(validation)} validation examples[/green]")

@app.command()
def train(config: Path) -> None:
    """Fine-tune the configured weak baseline with DPO."""
    from .trainers import PreferenceTrainer, TrainingConfig
    cfg = load_config(config)
    training = cfg["training"]
    trainer = PreferenceTrainer(TrainingConfig(method=training["method"], beta=training["beta"], lambda_orpo=training["lambda_orpo"], max_length=training["max_length"], batch_size=training["batch_size"]))
    output_dir = Path(cfg["paths"]["output_dir"]) / "dpo_model"
    result = trainer.train(cfg["paths"]["train_data"], output_dir, cfg["paths"].get("model_name", "distilgpt2"))
    print(f"[green]Training complete: {result}[/green]")

@app.command(name="evaluate-model")
def evaluate_model(config: Path) -> None:
    """Evaluate the saved DPO model on validation data."""
    evaluate(config)


@app.command()
def regression(config: Path) -> None:
    """Run real before/after generation checks."""
    import json
    cfg = load_config(config)
    prompt_file = Path(cfg["evaluation"]["regression_prompts"])
    prompts = [line.split(".", 1)[1].strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()[:1].isdigit() and "." in line]
    if not prompts:
        raise typer.BadParameter(f"no numbered prompts found in {prompt_file}")
    tuned_model = Path(cfg["paths"]["output_dir"]) / "dpo_model"
    if not tuned_model.exists():
        raise typer.BadParameter(f"trained model not found: {tuned_model}; run pref-lab train first")
    before_torch, before_tokenizer, before = load_causal_model(cfg["paths"].get("model_name", "distilgpt2"))
    after_torch, after_tokenizer, after = load_causal_model(tuned_model)
    results = [{"prompt": prompt, "before": generate_response(before_torch, before_tokenizer, before, prompt), "after": generate_response(after_torch, after_tokenizer, after, prompt)} for prompt in prompts]
    out = Path(cfg["paths"]["output_dir"]) / "regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[green]Wrote real before/after regression outputs to {out}[/green]")

if __name__ == "__main__":
    app()
