PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = experiment
PYTHON_INTERPRETER = python3
DATA_REPO=https://gitlab2.informatik.uni-wuerzburg.de/de_uniwue_ethikchat/ethikchat_data.git
DATA_REPO_TAG=1.0.0
SHELL := /bin/bash
.DEFAULT_GOAL := all
GITLAB_PERSONAL_ACCESS_TOKEN=$(shell cat .gitlab_personal_access_token)


venv/bin/activate: requirements.txt
	test -d venv || $(PYTHON_INTERPRETER) -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

.PHONY: requirements
## Install Python Dependencies
requirements: venv/bin/activate

freeze:
	source venv/bin/activate && pip list --format=freeze > requirements.txt

.PHONY: data

data:
	mkdir -p data/ethikchat_data
	curl -L -o data/ethikchat_data.zip --header "PRIVATE-TOKEN: $(GITLAB_PERSONAL_ACCESS_TOKEN)" https://gitlab2.informatik.uni-wuerzburg.de/de_uniwue_ethikchat/ethikchat_data/-/archive/1.0.0/ethikchat_data-1.0.0.zip
	unzip data/ethikchat_data.zip -d data/ethikchat_data
	rm data/ethikchat_data.zip


.PHONY: all
all:
	$(MAKE) requirements
	$(MAKE) data

.PHONY: clean
clean:
	rm -rf venv
	rm -rf data/de.uniwue.ethik-chat.Data
