FROM nvcr.io/nvidia/tensorrt:23.12-py3

WORKDIR /ml-project

RUN pip install uv && \
    apt-get update && apt-get install -y --no-install-recommends git bash make

COPY pyproject.toml uv.lock ./

RUN uv sync --no-cache

COPY . .

ENV PYTHONPATH /ml-project

EXPOSE 8080
