# Data Card

See the completed dataset documentation in this file. This card describes the repository-provided preference data, its schema, split method, limitations, and safety/PII review.

- **Dataset**: Preference Alignment Lab sample preferences
- **Source**: `data/sample_preferences.jsonl`
- **Schema**: JSONL with `prompt`, `chosen`, `rejected`, and optional `metadata`
- **Labeling**: `chosen` is intended to be more accurate/helpful than `rejected`
- **Known limitations**: Small, English-only educational dataset focused on introductory ML topics
- **PII/safety review**: No PII was observed during manual review; this is not a guarantee of safety for arbitrary inputs
- **Split**: Deterministic train/validation split by normalized prompt using seed 42; no separate test split
