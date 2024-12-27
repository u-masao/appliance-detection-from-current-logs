#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = appliance-detection-from-current-logs
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## dvc repro
.PHONY: repro
repro:
	uv run dvc repro


## formatting and lint
.PHONY: lint
LINT_TARGET=src tests
lint:
	uv run isort $(LINT_TARGET)
	uv run black -l 79 $(LINT_TARGET)
	uv run flake8 $(LINT_TARGET)

.PHONY: test
test:
	PYTHONPATH=. uv run pytest -s tests

## visualize
.PHONY: visualize
visualize:
	PYTHONPATH=. uv run gradio src/visualize.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
