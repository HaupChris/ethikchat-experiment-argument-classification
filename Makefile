.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = experiment
PYTHON_INTERPRETER = python3
DATA_REPO=https://gitlab2.informatik.uni-wuerzburg.de
DATA_REPO_TAG=1.0.0
SHELL := /bin/bash

.PHONY: requirements
## Install Python Dependencies
requirements:
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

freeze:
	pip list --format=freeze > requirements.txt

data:
	git clone $(DATA_REPO) data/de.uniwue.ethik-chat.Data
	cd data/de.uniwue.ethik-chat.Data && git checkout tags/$(DATA_REPO_TAG)

all: install data