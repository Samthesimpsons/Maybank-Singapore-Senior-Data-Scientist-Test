#!/bin/bash

# Run flake8
echo "--------- Linting with flake8 ---------"
poetry run flake8 --ignore=E203,E266,W503,E501 --max-line-length=88 src

# Run ruff
echo "--------- Running with ruff ---------"
poetry run ruff check src
