FROM python:3.12-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install dependencies
CMD apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY . /root/workspaces/
