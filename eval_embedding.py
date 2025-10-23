#!/usr/bin/env python3
"""
eval_embedding.py - Evaluate an embedding-based classifier.

Usage:
    python eval_embedding.py --config configs/config_coarse_embedding.yaml --checkpoint models/model_coarse_embedding.pkl
"""

import argparse
import csv
from typing import List, Tuple

import yaml
import numpy as np

from models.embedding_classifier import EmbeddingClassifier


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
    parser = argparse.ArgumentParser(description="Evaluate embedding classifier")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    classifier = EmbeddingClassifier.load(args.checkpoint)
    
    # Load validation data
    print(f"Loading validation data from: {cfg['data']['valid_path']}")
    valid_texts, valid_labels = load_data(
        cfg['data']['valid_path'],
        cfg['data']['label_column']
    )
    print(f"  Loaded {len(valid_texts):,} validation examples")
    
    # Predict
    print(f"\nRunning predictions...")
    batch_size = cfg.get('embedding', {}).get('batch_size', 32)
    predictions = classifier.predict(valid_texts, batch_size=batch_size)
    
    # Calculate accuracy
    correct = np.sum(predictions == valid_labels)
    accuracy = correct / len(valid_labels)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Correct:         {correct:,}")
    print(f"Total:           {len(valid_labels):,}")
    print("="*60)


if __name__ == "__main__":
    main()

