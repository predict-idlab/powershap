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