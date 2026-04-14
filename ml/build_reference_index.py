#!/usr/bin/env python3
"""
Build a FAISS reference embedding index for retrieval-based LEGO part identification.

Instead of classifying into a fixed set of N parts (which requires retraining when
new parts are added), this script:

  1. Loads a trained contrastive encoder (ONNX or PyTorch checkpoint)
  2. Runs all renders / catalog images for each part through the encoder
  3. Averages embeddings per part → one representative vector per class
  4. Builds a FAISS flat L2 index over all class embeddings
  5. Saves: index.faiss, labels.json, stats.json

At inference time, the backend embeds a query image and calls index.search(query, k=5)
to retrieve the top-5 matching part numbers — no retraining needed to add new parts.

Usage:
    # From renders directory (after batch rendering):
    python build_reference_index.py \
        --checkpoint  ./ml/output/contrastive/20250413/exports/contrastive_encoder.onnx \
        --data-dir    ./ml/data/renders \
        --output-dir  ./backend/models/index

    # From mixed dataset (renders + downloaded real images):
    python build_reference_index.py \
        --checkpoint  ./ml/output/contrastive/20250413/checkpoints/checkpoint_epoch_099.pt \
        --data-dir    ./ml/data/splits/train \
        --output-dir  ./backend/models/index \
        --max-per-class 50 \
        --device cuda

    # Quick test with just the first 20 parts:
    python build_reference_index.py --checkpoint model.onnx --data-dir ./data/renders \
        --output-dir ./index_test --max-parts 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Install with: pip install faiss-cpu  (or faiss-gpu)")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torchvision.transforms as T
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# ============================================================================
# Image Preprocessing
# ============================================================================

def get_transform(image_size: int = 224):
    """Standard ImageNet normalization transform."""
    if not TV_AVAILABLE:
        raise ImportError("torchvision required: pip install torchvision")
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: Path, transform, image_size: int = 224) -> Optional[np.ndarray]:
    """Load an image and return as numpy array [1, 3, H, W]."""
    try:
        img = Image.open(path).convert('RGB')
        tensor = transform(img)          # [3, H, W]
        return tensor.unsqueeze(0).numpy()  # [1, 3, H, W]
    except Exception as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return None


# ============================================================================
# Encoder Loaders
# ============================================================================

class OnnxEncoder:
    """Wraps an ONNX Runtime session for embedding inference."""

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"  ONNX model loaded: {onnx_path}")
        print(f"  Provider: {self.session.get_providers()[0]}")

    def embed(self, images: np.ndarray) -> np.ndarray:
        """images: [B, 3, H, W] float32. Returns [B, embed_dim] normalized."""
        outputs = self.session.run(None, {self.input_name: images.astype(np.float32)})
        embeddings = outputs[0]
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)


class TorchEncoder:
    """Wraps a saved ContrastiveModel PyTorch checkpoint for embedding inference."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        if not TORCH_AVAILABLE:
            raise ImportError("torch required: pip install torch")

        # Import the model class from train_contrastive.py
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from train_contrastive import ContrastiveModel
        except ImportError as e:
            raise ImportError(f"Could not import ContrastiveModel: {e}")

        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Reconstruct model — try to read args from checkpoint
        saved_args = checkpoint.get('args', {})
        use_lora = saved_args.get('no_lora') is False  # default: LoRA was on

        self.model = ContrastiveModel(
            backbone_name='vit_base_patch14_dinov2',
            projection_dim=128,
            use_lora=use_lora,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"  PyTorch checkpoint loaded: {checkpoint_path}")
        print(f"  Device: {device}")

    def embed(self, images: np.ndarray) -> np.ndarray:
        """images: [B, 3, H, W] float32. Returns [B, embed_dim] normalized."""
        with torch.no_grad():
            t = torch.from_numpy(images).to(self.device)
            emb = self.model(t)
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()


def load_encoder(checkpoint: str, device: str = 'cpu'):
    """Auto-detect checkpoint type and return appropriate encoder."""
    if checkpoint.endswith('.onnx'):
        return OnnxEncoder(checkpoint)
    elif checkpoint.endswith('.pt') or checkpoint.endswith('.pth'):
        return TorchEncoder(checkpoint, device=device)
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint}. Expected .onnx or .pt")


# ============================================================================
# Index Building
# ============================================================================

def collect_images_by_class(
    data_dir: Path,
    max_per_class: int = 100,
    max_parts: Optional[int] = None,
) -> Dict[str, List[Path]]:
    """
    Scan data_dir for class subdirectories.
    Returns {class_name: [image_path, ...]} with at most max_per_class images per class.
    """
    images_by_class = {}
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if max_parts:
        class_dirs = class_dirs[:max_parts]

    for class_dir in class_dirs:
        images = (list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
        if not images:
            continue
        # Prefer diverse angles: sample evenly if more than max_per_class
        if len(images) > max_per_class:
            step = len(images) // max_per_class
            images = images[::step][:max_per_class]
        images_by_class[class_dir.name] = images

    return images_by_class


def build_class_embeddings(
    images_by_class: Dict[str, List[Path]],
    encoder,
    transform,
    image_size: int = 224,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """
    Embed all images for each class, average per class, return:
      embeddings: [N_classes, embed_dim]
      labels:     [N_classes]  (part numbers as strings)
    """
    class_names = sorted(images_by_class.keys())
    class_embeddings = []
    valid_labels = []

    iter_classes = tqdm(class_names, desc="Embedding classes") if TQDM else class_names

    for class_name in iter_classes:
        image_paths = images_by_class[class_name]
        class_embs = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_arrays = []
            for p in batch_paths:
                arr = load_image(p, transform, image_size)
                if arr is not None:
                    batch_arrays.append(arr)

            if not batch_arrays:
                continue

            batch = np.concatenate(batch_arrays, axis=0)  # [B, 3, H, W]
            embs = encoder.embed(batch)                    # [B, embed_dim]
            class_embs.append(embs)

        if not class_embs:
            continue

        # Average all embeddings for this class → one representative vector
        all_embs = np.concatenate(class_embs, axis=0)  # [N_images, embed_dim]
        mean_emb = all_embs.mean(axis=0)
        # Re-normalize after averaging
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        class_embeddings.append(mean_emb)
        valid_labels.append(class_name)

    embeddings = np.stack(class_embeddings, axis=0).astype(np.float32)
    return embeddings, valid_labels


def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool = False,
) -> "faiss.Index":
    """
    Build a FAISS flat inner-product index (equivalent to cosine similarity
    for normalized vectors).

    FlatIP is the simplest index: exact search, no approximation.
    For >50K classes, consider IndexIVFFlat for faster search.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss required: pip install faiss-cpu")

    embed_dim = embeddings.shape[1]
    print(f"  Building FAISS IndexFlatIP ({embeddings.shape[0]} vectors, dim={embed_dim})")

    index = faiss.IndexFlatIP(embed_dim)  # Inner product = cosine sim for L2-normalized

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  Using GPU-accelerated FAISS index")
        except Exception as e:
            print(f"  GPU FAISS failed ({e}), falling back to CPU")

    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors")
    return index


def query_index(
    index: "faiss.Index",
    labels: List[str],
    query_embedding: np.ndarray,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Search the index for the top-k nearest parts.
    Returns [(part_num, similarity_score), ...] sorted by score descending.
    """
    query = query_embedding.astype(np.float32).reshape(1, -1)
    # Re-normalize
    query = query / (np.linalg.norm(query) + 1e-8)

    distances, indices = index.search(query, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(labels):
            continue
        results.append((labels[idx], float(dist)))
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS reference index for retrieval-based LEGO part ID"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model (.onnx or .pt checkpoint)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with class subdirs: data_dir/{part_num}/*.png')
    parser.add_argument('--output-dir', type=str, default='./backend/models/index',
                        help='Output directory for index files (default: ./backend/models/index)')
    parser.add_argument('--max-per-class', type=int, default=100,
                        help='Max images per class to average (default: 100)')
    parser.add_argument('--max-parts', type=int, default=None,
                        help='Limit number of parts (for testing)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size matching training (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Embedding batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device for .pt checkpoints (default: cpu)')
    parser.add_argument('--gpu-index', action='store_true',
                        help='Use GPU-accelerated FAISS index')
    parser.add_argument('--test-query', type=str, default=None,
                        help='Path to a test image to query against the built index')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BrickScan — FAISS Reference Index Builder")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {output_dir}")

    # 1. Load encoder
    print("\n[1/4] Loading encoder...")
    encoder = load_encoder(args.checkpoint, device=args.device)
    transform = get_transform(args.image_size)

    # 2. Collect images
    print(f"\n[2/4] Collecting images from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    images_by_class = collect_images_by_class(
        data_dir,
        max_per_class=args.max_per_class,
        max_parts=args.max_parts,
    )
    total_images = sum(len(v) for v in images_by_class.values())
    print(f"  Found {len(images_by_class)} classes, {total_images} images total")

    if not images_by_class:
        print("ERROR: No images found. Check --data-dir.")
        sys.exit(1)

    # 3. Build embeddings
    print(f"\n[3/4] Computing class embeddings...")
    t0 = time.time()
    embeddings, labels = build_class_embeddings(
        images_by_class, encoder, transform,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f"  Embedded {len(labels)} classes in {elapsed:.1f}s")
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Build FAISS index
    print(f"\n[4/4] Building FAISS index...")
    index = build_faiss_index(embeddings, use_gpu=args.gpu_index)

    # Save index
    index_path = output_dir / "index.faiss"
    labels_path = output_dir / "labels.json"
    stats_path  = output_dir / "index_stats.json"

    faiss.write_index(index, str(index_path))
    print(f"  Saved index: {index_path}")

    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"  Saved labels: {labels_path} ({len(labels)} parts)")

    stats = {
        "n_parts":       len(labels),
        "embed_dim":     int(embeddings.shape[1]),
        "max_per_class": args.max_per_class,
        "image_size":    args.image_size,
        "data_dir":      str(args.data_dir),
        "checkpoint":    str(args.checkpoint),
        "built_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {stats_path}")

    # Optional: test query
    if args.test_query:
        print(f"\n--- Test Query: {args.test_query} ---")
        query_arr = load_image(Path(args.test_query), transform, args.image_size)
        if query_arr is not None:
            query_emb = encoder.embed(query_arr)
            results = query_index(index, labels, query_emb, k=5)
            print(f"  Top-5 matches:")
            for i, (part_num, score) in enumerate(results):
                print(f"    {i+1}. Part {part_num}  (similarity: {score:.4f})")

    print("\n" + "=" * 60)
    print("Index built successfully!")
    print(f"  Load in inference with:")
    print(f"    import faiss, json")
    print(f"    index  = faiss.read_index('{index_path}')")
    print(f"    labels = json.load(open('{labels_path}'))")
    print(f"    D, I   = index.search(query_embedding, k=5)")
    print(f"    top_k  = [(labels[i], float(d)) for d, i in zip(D[0], I[0])]")
    print("=" * 60)


if __name__ == '__main__':
    main()
