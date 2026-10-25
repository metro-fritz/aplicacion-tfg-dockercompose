"""FastAPI inference service. Interactive docs at /docs.

Endpoints
  GET  /health    -> liveness/readiness (used by docker compose)
  GET  /models    -> model metadata for the UI
  POST /classify  -> multipart form: `image` (file) + `model_id` (text)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from classifier import ImageClassifier, InvalidImageError
from registry import MODELS


# ── Response schemas (also documented automatically in /docs) ─────────────
class Prediction(BaseModel):
    rank: int
    label: str
    class_id: int
    probability: float


class ClassificationResult(BaseModel):
    model_id: str
    predictions: list[Prediction]
    top1_label: str
    top1_prob: float


# ── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load every model before accepting requests.
    app.state.classifier = ImageClassifier(MODELS)
    yield


app = FastAPI(title="TFG inference service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, dict]:
    return {model_id: spec.metadata() for model_id, spec in MODELS.items()}


# Plain `def` (not async): FastAPI runs it in a thread pool, so the blocking
# PyTorch call doesn't freeze the event loop.
@app.post("/classify", response_model=ClassificationResult)
def classify(image: UploadFile = File(...), model_id: str = Form("vgg")):
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo desconocido: {model_id}")

    try:
        return app.state.classifier.classify(image.file.read(), model_id)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
