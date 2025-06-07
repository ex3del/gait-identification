#!/bin/bash
set -e # Останавливать выполнение при любой ошибке

echo ">>> Running feature baking for training..."
uv run python -m pose.mvp.feature_bake

echo ">>> Running feature baking for evaluation..."
uv run python -m pose.mvp.feature_bake preprocessing=eval

echo ">>> Running LSTM model training..."
uv run python -m pose.mvp.models.LSTM.LSTM

echo ">>> Pipeline finished successfully!"
