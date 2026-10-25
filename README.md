# Aplicación TFG — Keratitis classification demo

Two containers:

| Service     | Stack             | Role                                                      |
|-------------|-------------------|-----------------------------------------------------------|
| `webapp`    | Flask + gunicorn  | Serves the UI, validates uploads, forwards to `inference` |
| `inference` | FastAPI + PyTorch | Loads the 4 models at start-up and runs predictions       |

Images travel in memory (multipart request webapp → inference); nothing is written to disk.

## Run

```bash
docker compose up --build
```

Open http://localhost:5000. The webapp waits until the inference service is healthy (models loaded).

To try the inference API directly (Swagger UI at http://localhost:8000/docs), uncomment the
`ports` section of `inference` in `docker-compose.yml`.

## Layout

```
inference/
  main.py            FastAPI routes: /health, /models, /classify
  classifier.py      pre-processing + inference on raw bytes
  registry.py        model catalogue (metadata + weights + constructor)
  architectures/     network definitions (VGG, ResNet, DenseNet, ViT)
  weights/           .pth files
webapp/
  main.py            local entry point (gunicorn in Docker)
  controller/        Flask routes
  services/          HTTP client for the inference service
  view/              templates + static (unchanged front end)
```

## Adding a model

Put the class in `inference/architectures/`, the weights in `inference/weights/`, and add one
`ModelSpec` entry to `MODELS` in `inference/registry.py`. The UI picks it up automatically.

## Local development (without Docker)

```bash
# terminal 1
cd inference
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --port 8000

# terminal 2  (INFERENCE_URL defaults to http://localhost:8000)
cd webapp
pip install -r requirements.txt
python main.py
```
