@echo off
setlocal

:: Скрипт для управления MLflow сервером
:: Проверяем, был ли передан аргумент (start или stop)
if "%1"=="" (
    echo Usage: %0 [start^|stop]
    exit /b 1
)

:: Обработка команды START
if /i "%1"=="start" (
    echo ^>^>^> Checking for virtual environment...
    if not exist .venv\Scripts\activate.bat (
        echo [ERROR] Virtual environment not found. Please run 'setup.bat' first.
        exit /b 1
    )

    echo ^>^>^> Starting MLflow server in a new window...
    :: 'start' запускает команду в новом окне и не блокирует текущий терминал.
    :: "MLflow Server" - это заголовок для нового окна.
    start "MLflow Server" uv run mlflow server --host 127.0.0.1 --port 8080

    echo.
    echo MLflow server started in a new window.
    echo To stop it, run: %0 stop
    exit /b 0
)

:: Обработка команды STOP
if /i "%1"=="stop" (
    echo ^>^>^> Attempting to stop the MLflow server...
    :: Ищем и принудительно завершаем процесс по заголовку его окна.
    :: /F - принудительное завершение. /FI - фильтр.
    taskkill /F /FI "WINDOWTITLE eq MLflow Server*" /T >nul 2>&1

    if %errorlevel% equ 0 (
        echo MLflow server was found and has been stopped.
    ) else (
        echo MLflow server process not found. It might already be stopped.
    )
    exit /b 0
)

:: Если команда не 'start' и не 'stop'
echo [ERROR] Unknown command: %1
echo Usage: %0 [start^|stop]
exit /b 1

endlocal
