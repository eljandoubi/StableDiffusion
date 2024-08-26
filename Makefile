# Use bash for all commands
SHELL := /bin/bash

# Path to Conda base directory
CONDA_BASE := $(shell conda info --base)

# Env args
ENV_NAME=SDenv

build:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda create --name $(ENV_NAME) python=3.12 -y && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt

pytest:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	pytest test_code.py

pylint:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	pylint **/*.py

check:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	pylint **/*.py && \
	pytest test_code.py

run:
	echo "The code is not completed yet."

clean:
	conda remove --name $(ENV_NAME) --all -y