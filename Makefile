#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = appliance-detection-from-current-logs
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python
STORAGE_PATH=/content/drive/MyDrive/dataset/appliance-detection-from-current-logs/files4

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## dvc repro
.PHONY: repro
repro: check_commit
	uv run dvc repro train_model convert_to_csv
	git commit dvc.lock -m 'run dvc repro'

.PHONY: check_commit
check_commit:
	git status
	git diff --exit-code
	git diff --exit-code --cached

### mlflow ui
.PHONY: mlflow_ui
mlflow_ui:
	uv run mlflow ui -h 0.0.0.0

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
	uv run dvc repro -s -f train_model
	git commit dvc.lock -m 'run dvc repro'

## sync_to_storage
ARCHIVES_DIR=archives
.PHONY: sync_to_storage
sync_to_storage:
	mkdir -p $(ARCHIVES_DIR)
	tar czf $(ARCHIVES_DIR)/data.tgz data/
	rsync -av $(ARCHIVES_DIR)/data.tgz $(STORAGE_PATH)/
	tar czf $(ARCHIVES_DIR)/mlruns.tgz mlruns/
	rsync -av $(ARCHIVES_DIR)/mlruns.tgz $(STORAGE_PATH)/
	tar czf $(ARCHIVES_DIR)/models.tgz models/
	rsync -av $(ARCHIVES_DIR)/models.tgz $(STORAGE_PATH)/
	uv run dvc push

## sync_from_storage
.PHONY: sync_from_storage
sync_from_storage:
	mkdir -p $(ARCHIVES_DIR)
	rsync -av $(STORAGE_PATH)/data.tgz $(ARCHIVES_DIR)/data.tgz 
	tar xzf $(ARCHIVES_DIR)/data.tgz
	rsync -av $(STORAGE_PATH)/mlruns.tgz $(ARCHIVES_DIR)/mlruns.tgz
	tar xzf $(ARCHIVES_DIR)/mlruns.tgz
	rsync -av $(STORAGE_PATH)/models.tgz $(ARCHIVES_DIR)/models.tgz
	tar xzf $(ARCHIVES_DIR)/models.tgz
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
