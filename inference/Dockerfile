FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch: the default PyPI wheel bundles CUDA (~several GB) that
# this service doesn't need. Installed in its own layer so it stays cached.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
