"""Catalogue of available models: display metadata + how to build each one.

To add a model: drop its architecture in `architectures/`, its weights in
`weights/`, and add one entry to MODELS. Nothing else needs to change.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from architectures import VGG, ResNet, DenseNet, VisionTransformer

WEIGHTS_DIR = Path(__file__).parent / "weights"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    params: str
    depth: int | str
    weights: str                           # file name inside WEIGHTS_DIR
    build: Callable[[], torch.nn.Module]   # returns an untrained instance

    def metadata(self) -> dict:
        """Public info sent to the web app (no loader / paths)."""
        return {"name": self.name, "description": self.description,
                "params": self.params, "depth": self.depth}

    def load(self) -> torch.nn.Module:
        """Build the network, load its weights on CPU and set eval mode."""
        model = self.build()
        state = torch.load(WEIGHTS_DIR / self.weights, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        return model.eval()


# Insertion order = order shown in the UI (the first one is pre-selected).
MODELS: dict[str, ModelSpec] = {
    "vgg": ModelSpec(
        name="VGG", description="VGG base", params="~ 33.000", depth=12,
        weights="vgg192x256.pth", build=VGG,
    ),
    "resnet": ModelSpec(
        name="ResNet", description="VGG base con conexiones residuales",
        params="~ 70.000", depth=12,
        weights="resnet256x192.pth", build=ResNet,
    ),
    "densenet": ModelSpec(
        name="DenseNet", description="VGG base con conexiones densas",
        params="~ 700.000", depth=15,
        weights="densenetCustom256x192.pth", build=lambda: DenseNet(k=8),
    ),
    "vit": ModelSpec(
        name="Vision Transformer",
        description="Transformador de vision con 4 cabezas y tamaño de patch de 8",
        params="~ 300.000", depth="3 bloques codificadores",
        weights="vit256x192.pth",
        build=lambda: VisionTransformer(patch_size=8, depth=3, embed_dim=64,
                                        num_heads=4, mlp_dim=256, dropout=0.2),
    ),
}
