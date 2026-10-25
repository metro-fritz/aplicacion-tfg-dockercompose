"""Thin HTTP client for the inference service. The only place that knows its URL/API."""
import os

import requests

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = (3, 30)  # (connect, read) seconds


class InferenceError(Exception):
    """Inference service unreachable or returned an error."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class InferenceClient:
    def __init__(self, base_url: str = INFERENCE_URL):
        self.base_url = base_url
        self._models: dict | None = None  # metadata is static -> cached after first fetch

    def models(self) -> dict:
        if self._models is None:
            self._models = self._request("GET", "/models")
        return self._models

    def classify(self, image_bytes: bytes, filename: str, content_type: str, model_id: str) -> dict:
        # Image goes straight from memory into a multipart body: no temp files.
        files = {"image": (filename, image_bytes, content_type)}
        return self._request("POST", "/classify", files=files, data={"model_id": model_id})

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            res = requests.request(method, self.base_url + path, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as e:
            raise InferenceError("El servicio de inferencia no está disponible.", 503) from e

        if not res.ok:
            # FastAPI puts error messages under "detail".
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise InferenceError(str(detail), res.status_code)
        return res.json()
