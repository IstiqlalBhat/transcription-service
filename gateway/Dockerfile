FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./gateway/

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD uvicorn gateway.main:app --host 0.0.0.0 --port $PORT
