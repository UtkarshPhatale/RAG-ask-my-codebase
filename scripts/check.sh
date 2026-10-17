#!/usr/bin/env bash
# One command for everything that must stay true before a push:
#   1. the full test suite (RLS adversarial, endpoints, query_logs, concurrency)
#   2. the eval question set is valid against the live database
#   3. the retrieval eval, which exits non-zero on ANY leak
# Runs against the live Supabase project in .env and makes a few real Claude calls, so it
# takes a couple of minutes. Needs the venv active. Stops at the first failure.
set -euo pipefail

[ -n "${VIRTUAL_ENV:-}" ] || { echo "activate the venv first: source venv/bin/activate"; exit 1; }
cd "$(dirname "$0")/.."

echo "== 1/3 tests"
python -m pytest -q 2>&1 | tail -3

echo "== 2/3 eval question set vs live database"
python -m scripts.validate_eval | tail -4

echo "== 3/3 retrieval eval (fails on any leak)"
python -m scripts.eval_retrieval | grep -E "^(contractor|senior_engineer) +[0-9]|LEAK"

echo "== all checks passed"
