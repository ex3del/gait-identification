# Настройка Docker для ML проекта

Этот каталог содержит `Dockerfile` и `docker-compose.yaml` для запуска проекта в контейнеризованной среде.

## Предварительные требования

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) для поддержки GPU.

## Использование

### Сборка образов

Для сборки Docker образов для сервисов выполните следующую команду из корневой директории проекта:

```bash
docker-compose -f docker/docker-compose.yaml build
```

### Запуск сервисов

Для запуска сервисов `app` и `mlflow` выполните:

```bash
docker-compose -f docker/docker-compose.yaml up -d
```

Флаг `-d` запускает контейнеры в фоновом режиме.

### Доступ к интерфейсу MLflow

Интерфейс сервера отслеживания MLflow будет доступен по адресу [http://localhost:8080](http://localhost:8080).

Ваши эксперименты и артефакты будут логироваться здесь.

### Запуск скриптов в контейнере

Для запуска ваших Python скриптов, сначала необходимо получить оболочку внутри контейнера `app`:

```bash
docker exec -it ml-project-app /bin/bash
```

Находясь внутри контейнера, вы можете запускать свои скрипты как обычно. Например:

```bash
python pose/train.py --config-name=vitpose_small_coco.yaml
```

Убедитесь, что ваш код использует переменную окружения `MLFLOW_TRACKING_URI` для подключения к серверу MLflow. Она уже настроена для вас в файле `docker-compose.yaml`.

### Остановка сервисов

Для остановки и удаления контейнеров, сетей и томов выполните:

```bash
docker-compose -f docker/docker-compose.yaml down
```
