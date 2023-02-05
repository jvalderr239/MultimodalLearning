# define the name of the virtual environment directory
VENV := venv
BIN := ${VENV}/bin
PYTHON := ${BIN}/python3
PIP := ${BIN}/pip
DIST := ${VENV}/dist

.DEFAULT_GOAL := help

define PRINT_HELP_SCRIPT
import re, sys

for line in sys.stdin:
	match = re.match('^([a-zA-Z_-]+):.*?##(.*)$$', line)
	if match:
		target, help = match.groups()
		print("%20s %s", (target, help))
endef

export PRINT_HELP_SCRIPT

python_src = src scripts/*.py models/* tests/*.py
coverage_src = src
# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	{PYTHON} -m venv $(VENV)
	${PIP} install --upgrade pip black isort mypy pytest coverage pylint
	${PIP} install -r requirements.txt

help:
	${PYTHON} -c "$$PRINT_HELP_SCRIPT" < "$(MAKEFILE_LIST)"

# venv is a shortcut target
venv: $(VENV)/bin/activate

check: venv check-format check-types lint

check-format:
	black --check ${python_src}
	isort --check-only ${python_src}

check-types:
	mypy ${python_src}

build: clean venv
	${PIP} install wheel
	${PYTHON} -m setup bdist_wheel --target ${VENV}

test: build
	${PIP} install ${DIST}/*
	coverage run --branch --source ${coverage_src} -m pytest src/tests
	coverage report -m 

coverage: test
	coverage html

format:
	black ${python_src}
	isort ${python_src}

lint:
	pylint ${python_src} -f parseable -r n

clean: clean-build clean-pyc clean-check clean-test

clean-build:
	rm -rf ${VENV}/
	rm -rf ${DIST}/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc: # remove python artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-check:
	find . -name '.mypy_cache' -exec rm -rf {} +

clean-test:
	find . -name '.pytest_cache' -exec rm -rf {} +
	rm -f coverage
	rm -rf htmlcov/

.PHONY: all venv clean build test help
