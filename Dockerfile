FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY agent/ /app/agent/

CMD ["python", "/app/agent/main.py"]
