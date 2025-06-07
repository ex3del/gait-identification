@echo off
setlocal

:: Этот скрипт предполагает, что виртуальное окружение уже создано
:: и MLflow сервер запущен (если он нужен для логирования).

:: Шаг 1: Feature baking для обучающих данных
echo ^>^>^> Running feature baking for training...
uv run python -m pose.mvp.feature_bake
if %errorlevel% neq 0 (
    echo [ERROR] Failed during feature baking for training.
    exit /b 1
)

:: Шаг 2: Feature baking для данных оценки
echo ^>^>^> Running feature baking for evaluation...
uv run python -m pose.mvp.feature_bake preprocessing=eval
if %errorlevel% neq 0 (
    echo [ERROR] Failed during feature baking for evaluation.
    exit /b 1
)

:: Шаг 3: Обучение модели LSTM
echo ^>^>^> Running LSTM model training...
uv run python -m pose.mvp.models.LSTM.LSTM
if %errorlevel% neq 0 (
    echo [ERROR] Failed during LSTM model training.
    exit /b 1
)

endlocal
echo.
echo ^>^>^> Pipeline finished successfully!
