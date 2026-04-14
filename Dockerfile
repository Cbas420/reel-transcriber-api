FROM python:3.9-slim

# Instalar FFmpeg y curl (curl lo usamos para descargar el video)
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Usar el puerto dinámico de Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}