#!/bin/bash
# Останавливать выполнение при любой ошибке
set -e

echo ">>> Starting model evaluation..."
uv run python -m pose.mvp.models.LSTM.eval
echo ">>> Evaluation complete."
