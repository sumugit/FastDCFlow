FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip && pip install pipenv