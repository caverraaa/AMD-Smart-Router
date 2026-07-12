FROM python:3.11-slim

RUN pip install --no-cache-dir llama-cpp-python==0.3.34 \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY models/gemma-2-2b-it-Q4_K_M.gguf /app/models/gemma-2-2b-it-Q4_K_M.gguf

COPY agent/ /app/agent/

ENV ENABLE_BATCHING=1

CMD ["python", "/app/agent/main.py"]
