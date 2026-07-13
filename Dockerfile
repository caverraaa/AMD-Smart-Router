FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY agent/ /app/agent/

# Accuracy recovery profile.  Experimental local lanes and factual batching
# remain in the source tree, but the submitted image does not enable them
# until they pass the external semantic judge on the complete golden set.
ENV ENABLE_BATCHING=0

CMD ["python", "/app/agent/main.py"]
