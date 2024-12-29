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
repro: check_commit
	uv run dvc repro
	git commit dvc.lock -m 'run dvc repro'

.PHONY: check_commit
check_commit:
	git status
	git diff --exit-code
	git diff --exit-code --cached

### mlflow ui
.PHONY: mlflow_ui
mlflow_ui:
	uv run mlflow ui

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

## train force
.PHONY: train
train:
	uv run dvc repro -s -f train_model inference

STORAGE_PATH=/content/drive/MyDrive/dataset/appliance-detection-from-current-logs/files
## sync_to_storage
.PHONY: sync_to_storage
sync_to_storage:
	rsync -a data $(STORAGE_PATH)/data/
	rsync -a mlflow $(STORAGE_PATH)/mlruns/
	rsync -a models $(STORAGE_PATH)/models/
	uv run dvc push

## sync_from_storage
.PHONY: sync_from_storage
sync_from_storage:
	rsync -a $(STORAGE_PATH)/data/ data/
	rsync -a $(STORAGE_PATH)/mlruns/ mlruns/
	rsync -a $(STORAGE_PATH)/models/ models/
	uv run dvc pull

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
