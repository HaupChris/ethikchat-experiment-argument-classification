# Base directory and project settings
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = experiment
PYTHON_INTERPRETER = python3
SHELL := /bin/bash
.DEFAULT_GOAL := all

# Repository settings
DATA_REPO=https://gitlab2.informatik.uni-wuerzburg.de/de_uniwue_ethikchat/ethikchat_data
# Default version, can be overwritten by specifying when invoking make
DATA_REPO_VERSION=main
GITLAB_PERSONAL_ACCESS_TOKEN=$(shell cat .gitlab_personal_access_token)

# Virtual environment setup
venv/bin/activate: requirements.txt
	test -d venv || $(PYTHON_INTERPRETER) -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

.PHONY: requirements
requirements: venv/bin/activate

freeze:
	source venv/bin/activate && pip list --format=freeze > requirements.txt

.PHONY: data
data:
	mkdir -p data/ethikchat_data
	curl -L -o data/ethikchat_data.zip --header "PRIVATE-TOKEN: $(GITLAB_PERSONAL_ACCESS_TOKEN)" "$(DATA_REPO)/-/archive/$(DATA_REPO_VERSION)/ethikchat_data-$(DATA_REPO_VERSION).zip"
	unzip data/ethikchat_data.zip -d data/ethikchat_data
	rm data/ethikchat_data.zip

.PHONY: all
all:
	$(MAKE) requirements
	$(MAKE) data

.PHONY: clean
clean:
	rm -rf venv
	rm -rf data/ethikchat_data
