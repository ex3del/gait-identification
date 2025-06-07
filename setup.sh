#!/bin/bash
set -e

echo ">>> Creating virtual environment..."
uv venv

echo ">>> Activating virtual environment..."
source .venv/bin/activate

echo ">>> Installing project in editable mode..."
uv pip install -e .

echo ">>> Pulling DVC data..."
dvc pull

echo ">>> Syncing development dependencies..."
uv sync --group dev

echo ">>> Installing pre-commit hooks..."
pre-commit install

echo ">>> Running all pre-commit hooks..."
pre-commit run -a

echo ">>> Setup complete!"
