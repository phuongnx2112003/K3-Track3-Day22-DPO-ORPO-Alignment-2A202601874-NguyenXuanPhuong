.PHONY: setup test lint typecheck format run-eval run-regression clean
setup:
	pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src tests
typecheck:
	mypy src
format:
	ruff format src tests
run-eval:
	pref-lab evaluate-model configs/local.yaml
run-regression:
	pref-lab regression configs/local.yaml
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs
