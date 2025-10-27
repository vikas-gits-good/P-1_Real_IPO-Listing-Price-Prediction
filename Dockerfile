FROM python:3.11.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli libgomp1 curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY requirements.txt ./

RUN uv pip sync --system requirements.txt

COPY . .

RUN playwright install

CMD ["python3", "app.py"]
