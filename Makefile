# Use bash for all commands
SHELL := /bin/bash

# Path to Conda base directory
CONDA_BASE := $(shell conda info --base)

# Env args
ENV_NAME=SDenv
REQUIREMENTS=requirements.txt

build:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda create --name $(ENV_NAME) python=3.12 -y && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt && \

clean:
	conda remove --name $(ENV_NAME) --all -y