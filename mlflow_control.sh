#!/bin/bash
# Скрипт для управления MLflow сервером

# Проверяем первый аргумент: start или stop
if [ "$1" == "start" ]; then
    echo ">>> Starting MLflow server in the background..."
    # Запускаем сервер в фоновом режиме и сохраняем его PID в файл
    uv run mlflow server --host 127.0.0.1 --port 8080 &
    echo $! > mlflow.pid
    echo "MLflow server started with PID $(cat mlflow.pid). Use './mlflow_control.sh stop' to terminate."

elif [ "$1" == "stop" ]; then
    if [ -f "mlflow.pid" ]; then
        PID=$(cat mlflow.pid)
        echo ">>> Stopping MLflow server with PID $PID..."
        kill $PID
        rm mlflow.pid
        echo "MLflow server stopped."
    else
        echo "MLflow server PID file not found. Is the server running?"
    fi
else
    echo "Usage: ./mlflow_control.sh [start|stop]"
fi
