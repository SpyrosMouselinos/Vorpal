#!/usr/bin/env python3
"""
train_embedding.py - Train an embedding-based classifier.

Usage:
    python train_embedding.py --config configs/config_coarse_embedding.yaml
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import yaml

from models.embedding_classifier import EmbeddingClassifier
from utils.embedding_cache import load_embeddings_cache, save_embeddings_cache


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data(filepath: str, label_column: str) -> Tuple[List[str], List[int]]:
    """Load data from CSV."""
    texts = []
    labels = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['caption'])
            labels.append(int(row[label_column]))
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Train embedding-based classifier")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    print("="*70)
    print("TRAINING EMBEDDING-BASED CLASSIFIER")
    print("="*70)
    
    # Load data
    print(f"\nLoading training data from: {cfg['data']['train_path']}")
    train_texts, train_labels = load_data(
        cfg['data']['train_path'],
        cfg['data']['label_column']
    )
    print(f"  Loaded {len(train_texts):,} training examples")
    
    # Check for cached embeddings
    encoder_name = cfg['model'].get('encoder', 'all-MiniLM-L6-v2')
    dataset_name = Path(cfg['data']['train_path']).stem
    use_cache = cfg.get('embedding', {}).get('cache_embeddings', True)
    
    # Initialize classifier
    classifier = EmbeddingClassifier(
        encoder_name=encoder_name,
        classifier_type=cfg['model'].get('classifier', 'logistic_regression'),
        classifier_params=cfg.get('classifier_params', {})
    )
    
    # Train
    batch_size = cfg.get('embedding', {}).get('batch_size', 32)
    
    if use_cache:
        print(f"\nChecking for cached embeddings...")
        cached = load_embeddings_cache(dataset_name, encoder_name)
        
        if cached is not None:
            embeddings, _ = cached
            print(f"Using cached embeddings (skipping encoding step)")
            
            # Train classifier directly
            print(f"\nTraining {classifier.classifier_type} classifier...")
            classifier._init_classifier(len(set(train_labels)))
            classifier.classifier.fit(embeddings, train_labels)
            print(f"✓ Training complete")
        else:
            # Train normally and cache
            classifier.train(train_texts, train_labels, batch_size=batch_size)
            
            # Cache the embeddings for future runs
            print(f"\nCaching embeddings for future runs...")
            embeddings = classifier.encode_texts(train_texts, batch_size=batch_size, show_progress=False)
            save_embeddings_cache(embeddings, dataset_name, encoder_name)
    else:
        classifier.train(train_texts, train_labels, batch_size=batch_size)
    
    # Save model
    checkpoint_path = cfg['training']['checkpoint_path']
    print(f"\nSaving model to: {checkpoint_path}")
    classifier.save(checkpoint_path)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Encoder: {encoder_name}")
    print(f"Classifier: {classifier.classifier_type}")
    print(f"\nNext step:")
    print(f"  python eval_embedding.py --config {args.config} --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()

