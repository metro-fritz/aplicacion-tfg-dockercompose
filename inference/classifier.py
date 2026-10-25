"""Pre-processing + inference. Works on raw image bytes: nothing touches disk."""
import io

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from registry import ModelSpec

LABELS = {0: "AK", 1: "FK", 2: "NSK", 3: "normal"}

# Same pre-processing used during training: 1-channel, 192x256.
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((192, 256)),
    transforms.ToTensor(),
])


class InvalidImageError(ValueError):
    """Raised when the received bytes are not a decodable image."""


class ImageClassifier:
    def __init__(self, specs: dict[str, ModelSpec]):
        # Models are small, so all of them are loaded once at start-up:
        # no first-request latency and the health check means "ready".
        self.models = {model_id: spec.load() for model_id, spec in specs.items()}

    @staticmethod
    def _to_tensor(image_bytes: bytes) -> torch.Tensor:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError("No se pudo decodificar la imagen.") from e
        return TRANSFORM(img).unsqueeze(0)  # add batch dim -> (1, 1, 192, 256)

    @torch.inference_mode()
    def classify(self, image_bytes: bytes, model_id: str, top_k: int = 4) -> dict:
        tensor = self._to_tensor(image_bytes)
        logits = self.models[model_id](tensor)

        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idx = probs.topk(min(top_k, probs.numel()))

        predictions = [
            {
                "rank": i + 1,
                "label": LABELS[idx.item()],
                "class_id": idx.item(),
                "probability": round(prob.item() * 100, 2),
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_idx))
        ]
        return {
            "model_id": model_id,
            "predictions": predictions,
            "top1_label": predictions[0]["label"],
            "top1_prob": predictions[0]["probability"],
        }
