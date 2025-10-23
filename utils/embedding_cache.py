"""
Utilities for caching embeddings to speed up training iterations.
"""

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def generate_cache_key(texts: List[str], encoder_name: str) -> str:
    """Generate a unique cache key for a set of texts and encoder."""
    # Hash the texts and encoder name
    text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()[:16]
    encoder_hash = hashlib.md5(encoder_name.encode()).hexdigest()[:8]
    return f"{encoder_hash}_{text_hash}"


def get_cache_path(dataset_name: str, encoder_name: str) -> Path:
    """Get the cache file path for a dataset."""
    cache_dir = Path('data/embedding_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize encoder name for filename
    encoder_safe = encoder_name.replace('/', '_').replace('-', '_')
    return cache_dir / f"{dataset_name}_{encoder_safe}.npy"


def save_embeddings_cache(
    embeddings: np.ndarray,
    dataset_name: str,
    encoder_name: str,
    metadata: Optional[dict] = None
):
    """
    Save embeddings to cache.
    
    Args:
        embeddings: numpy array of embeddings
        dataset_name: name of the dataset (e.g., 'train_combined')
        encoder_name: name of the encoder model
        metadata: optional metadata to save with embeddings
    """
    cache_path = get_cache_path(dataset_name, encoder_name)
    
    # Save embeddings
    np.save(cache_path, embeddings)
    
    # Save metadata if provided
    if metadata:
        meta_path = cache_path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    print(f"✓ Cached embeddings to: {cache_path}")
    print(f"  Shape: {embeddings.shape}, Size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")


def load_embeddings_cache(
    dataset_name: str,
    encoder_name: str
) -> Optional[Tuple[np.ndarray, Optional[dict]]]:
    """
    Load embeddings from cache if available.
    
    Returns:
        (embeddings, metadata) tuple if cache exists, None otherwise
    """
    cache_path = get_cache_path(dataset_name, encoder_name)
    
    if not cache_path.exists():
        return None
    
    # Load embeddings
    embeddings = np.load(cache_path)
    
    # Load metadata if exists
    meta_path = cache_path.with_suffix('.meta.pkl')
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
    
    print(f"✓ Loaded cached embeddings from: {cache_path}")
    print(f"  Shape: {embeddings.shape}")
    
    return embeddings, metadata


def clear_cache(dataset_name: Optional[str] = None, encoder_name: Optional[str] = None):
    """
    Clear embedding cache.
    
    Args:
        dataset_name: If provided, clear only this dataset
        encoder_name: If provided, clear only this encoder
    """
    cache_dir = Path('data/embedding_cache')
    
    if not cache_dir.exists():
        print("No cache directory found")
        return
    
    if dataset_name and encoder_name:
        # Clear specific cache
        cache_path = get_cache_path(dataset_name, encoder_name)
        if cache_path.exists():
            cache_path.unlink()
            meta_path = cache_path.with_suffix('.meta.pkl')
            if meta_path.exists():
                meta_path.unlink()
            print(f"✓ Cleared cache: {cache_path.name}")
    else:
        # Clear all cache
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared all embedding cache")

