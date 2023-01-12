isort = isort powershap tests
black = black powershap tests

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	ruff powershap tests
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: clean
clean:
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .ruff_cache
	rm -rf catboost_info
