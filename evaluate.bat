@echo off
setlocal

:: Шаг 1: Запуск скрипта оценки модели
echo ^>^>^> Starting model evaluation...
uv run python -m pose.mvp.models.LSTM.eval

:: Шаг 2: Проверка, успешно ли завершился скрипт
if %errorlevel% neq 0 (
    echo [ERROR] Model evaluation script failed with exit code %errorlevel%.
    exit /b 1
)

endlocal
echo.
echo ^>^>^> Evaluation complete.
