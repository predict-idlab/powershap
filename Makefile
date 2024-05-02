isort = poetry run isort powershap tests
black = poetry run black powershap tests

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	poetry run ruff powershap tests
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: clean
clean:
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .ruff_cache
	rm -rf catboost_info
