"""
BrickScan ML module — new model family for the hybrid recognition pipeline.

Exports:
  model_manager    — ModelManager singleton (contrastive encoder + distilled student + YOLO)
  embedding_library — EmbeddingLibrary singleton (k-NN index over known parts)
"""

from app.ml.model_manager import ModelManager, BoundingBox
from app.ml.embedding_library import EmbeddingLibrary

model_manager = ModelManager.get()
embedding_library = EmbeddingLibrary.get()

__all__ = ["ModelManager", "EmbeddingLibrary", "BoundingBox", "model_manager", "embedding_library"]
