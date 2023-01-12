isort = isort powershap tests
black = black powershap tests

.PHONY: format
format:
	$(isort)
	$(black)
