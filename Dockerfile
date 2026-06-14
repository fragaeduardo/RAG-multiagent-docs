FROM python:3.12-slim

WORKDIR /app

# ── Dependências de Sistema ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    make \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    xvfb \
    chromium \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Cria o usuário 'eduardo' (ou dev) atrelado ao UID 1000
RUN groupadd -g 1000 desenvolvedor && \
    useradd -u 1000 -g desenvolvedor -m -s /bin/bash eduardo

COPY requirements.txt .

# ── Dependências Python ────────────────────────────────────────────────────
# Instala dependencias do Python usando o cache persistente do Docker BuildKit
# Além disso, concedemos permissão na pasta de pacotes para o usuário 1000,
# permitindo que bibliotecas salvem pesos dinamicamente se necessário.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt && \
    chown -R 1000:1000 /usr/local/lib/python3.12/site-packages/

COPY . .

CMD ["tail", "-f", "/dev/null"]
