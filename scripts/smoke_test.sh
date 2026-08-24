#!/usr/bin/env bash
set -euo pipefail
.venv/bin/pref-lab validate data/sample_preferences.jsonl
.venv/bin/pref-lab evaluate-model configs/local.yaml
.venv/bin/pref-lab regression configs/local.yaml
cat outputs/metrics.json
