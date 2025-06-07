@echo off
setlocal

:: Шаг 1: Создание виртуального окружения
echo ^>^>^> Creating virtual environment...
uv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

:: Шаг 2: Активация окружения
echo ^>^>^> Activating the environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    exit /b 1
)

:: Шаг 3: Установка зависимостей и инструментов (теперь в активированном окружении)
echo ^>^>^> Installing project in editable mode...
uv pip install -e .
if %errorlevel% neq 0 (
    echo [ERROR] Failed 'uv pip install'.
    exit /b 1
)

echo ^>^>^> Pulling DVC data...
dvc pull
if %errorlevel% neq 0 (
    echo [ERROR] Failed 'dvc pull'.
    exit /b 1
)

echo ^>^>^> Syncing development dependencies...
uv sync --group dev
if %errorlevel% neq 0 (
    echo [ERROR] Failed 'uv sync'.
    exit /b 1
)

echo ^>^>^> Installing pre-commit hooks...
pre-commit install
if %errorlevel% neq 0 (
    echo [ERROR] Failed 'pre-commit install'.
    exit /b 1
)

echo ^>^>^> Running all pre-commit hooks...
pre-commit run -a
if %errorlevel% neq 0 (
    echo [ERROR] One of the pre-commit hooks failed.
    exit /b 1
)

endlocal
echo.
echo ^>^>^> Setup complete! Script finished successfully.
