#!/usr/bin/env python3
"""
inference_embedding.py - Run inference with embedding-based classifier.

Usage:
    # From stdin
    echo "text here" | python inference_embedding.py --config config.yaml --checkpoint model.pkl --stdin
    
    # From CSV
    python inference_embedding.py --config config.yaml --checkpoint model.pkl --input_csv data.csv --text_column caption
"""

import argparse
import csv
import sys

import yaml
import numpy as np

from models.embedding_classifier import EmbeddingClassifier


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Inference with embedding classifier")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    
    # Input sources
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--input_csv", help="Path to CSV with data")
    parser.add_argument("--text_column", default="caption", help="Column name for text")
    
    # Options
    parser.add_argument("--show_proba", action="store_true", help="Show probabilities")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    print(f"[load] Loading model from: {args.checkpoint}", file=sys.stderr)
    classifier = EmbeddingClassifier.load(args.checkpoint)
    
    # Process input
    if args.stdin:
        # Read from stdin
        print("[inference] Reading from stdin...", file=sys.stderr)
        
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            
            if args.show_proba:
                probs = classifier.predict_proba(text)
                pred = np.argmax(probs) + 1  # 1-indexed
                conf = probs[pred - 1]
                print(f"{pred}\t{conf:.4f}\t{text}")
            else:
                pred = classifier.predict(text)
                print(f"{pred}\t{text}")
    
    elif args.input_csv:
        # Read from CSV
        print(f"[inference] Reading from CSV: {args.input_csv}", file=sys.stderr)
        
        with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Collect all texts for batch processing
            texts = []
            for row in reader:
                texts.append(row[args.text_column])
            
            print(f"[inference] Processing {len(texts):,} examples...", file=sys.stderr)
            
            # Batch predict
            batch_size = cfg.get('embedding', {}).get('batch_size', 32)
            
            if args.show_proba:
                probs = classifier.predict_proba(texts, batch_size=batch_size)
                predictions = np.argmax(probs, axis=1) + 1  # 1-indexed
                confidences = np.max(probs, axis=1)
                
                print("predicted_label\tconfidence\ttext")
                for pred, conf, text in zip(predictions, confidences, texts):
                    print(f"{pred}\t{conf:.4f}\t{text}")
            else:
                predictions = classifier.predict(texts, batch_size=batch_size)
                
                print("predicted_label\ttext")
                for pred, text in zip(predictions, texts):
                    print(f"{pred}\t{text}")
        
        print(f"[done] Processed {len(texts):,} examples", file=sys.stderr)
    
    else:
        parser.error("Must specify either --stdin or --input_csv")


if __name__ == "__main__":
    main()

