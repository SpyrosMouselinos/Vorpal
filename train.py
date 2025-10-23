#!/usr/bin/env python3
"""
train_vw.py - Train a VW Next multiclass text classifier with progressive validation.

Usage:
    python train_vw.py --config config.example.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any

import yaml
import vowpal_wabbit_next as vw


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Clean text for VW format (remove reserved chars)."""
    return text.replace("|", " ").replace(":", " ").replace("\t", " ").strip()


def generate_ngrams(tokens, n=2):
    """Generate n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams


def to_vw_line(label: Optional[int], text: str, namespace: str = "t", ngram: int = 2) -> str:
    """Convert (label, text) to VW format line with n-grams."""
    txt = clean_text(text)
    tokens = txt.lower().split()
    
    # Generate features: unigrams + bigrams (or higher order n-grams)
    features = tokens.copy()
    if ngram >= 2 and len(tokens) >= 2:
        features.extend(generate_ngrams(tokens, 2))
    if ngram >= 3 and len(tokens) >= 3:
        features.extend(generate_ngrams(tokens, 3))
    
    feature_str = ' '.join(features)
    
    if label is None:
        return f"|{namespace} {feature_str}"
    else:
        return f"{int(label)} |{namespace} {feature_str}"


def iter_data(
    path: str,
    text_col: str,
    label_col: str,
    label_mapping: Optional[Dict[str, int]] = None
) -> Iterator[Tuple[int, str]]:
    """
    Yield (label_id, text) from CSV or Parquet.
    
    Supports both formats; detects by extension.
    """
    path_obj = Path(path)
    
    if path_obj.suffix == '.parquet':
        # Use PyArrow for streaming Parquet
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("Install pyarrow for Parquet support: pip install pyarrow")
        
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                label_raw = row[label_col]
                text = str(row[text_col])
                
                # Map label if needed
                if label_mapping and isinstance(label_raw, str):
                    label_id = label_mapping[label_raw]
                else:
                    label_id = int(label_raw)
                
                yield label_id, text
    
    elif path_obj.suffix == '.csv':
        # Use CSV module for streaming
        import csv
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_raw = row[label_col]
                text = str(row[text_col])
                
                # Map label if needed
                if label_mapping and isinstance(label_raw, str):
                    label_id = label_mapping[label_raw]
                else:
                    label_id = int(label_raw)
                
                yield label_id, text
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}")


def train_progressive(
    ws: vw.Workspace,
    parser: vw.TextFormatParser,
    data_iter: Iterator[Tuple[int, str]],
    namespace: str,
    ngram: int,
    log_every: int = 200000
) -> float:
    """
    Train with progressive validation (predict-then-learn).
    
    Returns:
        Overall progressive accuracy
    """
    pred_type = ws.prediction_type
    seen = correct = 0
    
    for i, (y, text) in enumerate(data_iter, 1):
        ex = parser.parse_line(to_vw_line(y, text, namespace, ngram))
        pred = ws.predict_then_learn_one(ex)
        
        # Convert prediction to class index (1..K)
        if pred_type.name == "Scalars":
            # List of class probabilities
            yhat = 1 + max(range(len(pred)), key=lambda k: pred[k])
        elif pred_type.name == "Multiclass":
            yhat = int(pred)
        else:
            raise RuntimeError(f"Unexpected prediction type: {pred_type}")
        
        correct += (yhat == y)
        seen += 1
        
        if i % log_every == 0:
            print(f"[train] examples={seen:,} accuracy={correct/seen:.4f}")
    
    final_acc = correct / max(seen, 1)
    print(f"[train] FINAL progressive accuracy: {final_acc:.4f}")
    return final_acc


def main():
    parser = argparse.ArgumentParser(description="Train VW Next classifier")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Build VW arguments (only supported options in vowpal-wabbit-next)
    vw_args = [
        "--oaa", str(cfg['model']['num_classes']),
        "--loss_function", cfg['vw']['loss_function'],
        "-b", str(cfg['vw']['bits']),
        "--learning_rate", str(cfg['vw']['learning_rate']),
        "--l2", str(cfg['vw']['l2']),
    ]
    
    if cfg['vw'].get('probabilities', False):
        vw_args.append("--probabilities")
    
    print(f"[init] Creating VW workspace with args: {' '.join(vw_args)}")
    print(f"[init] N-gram order: {cfg['vw'].get('ngram', 2)} (generated in text features)")
    ws = vw.Workspace(vw_args)
    text_parser = vw.TextFormatParser(ws)
    
    # Train
    print(f"[init] Loading training data from: {cfg['data']['train_path']}")
    train_iter = iter_data(
        cfg['data']['train_path'],
        cfg['data']['text_column'],
        cfg['data']['label_column'],
        cfg['data'].get('label_mapping')
    )
    
    train_acc = train_progressive(
        ws,
        text_parser,
        train_iter,
        cfg['model']['namespace'],
        cfg['vw'].get('ngram', 2),
        cfg['training']['log_every']
    )
    
    # Save model
    checkpoint_path = cfg['training']['checkpoint_path']
    print(f"[save] Saving model to: {checkpoint_path}")
    with open(checkpoint_path, 'wb') as f:
        f.write(ws.serialize())
    
    # Optionally save label mapping
    if cfg['training'].get('save_labels') and cfg['data'].get('label_mapping'):
        labels_path = checkpoint_path.replace('.vw', '.labels.json').replace('.vwbin', '.labels.json')
        with open(labels_path, 'w') as f:
            json.dump(cfg['data']['label_mapping'], f, indent=2)
        print(f"[save] Saved label mapping to: {labels_path}")
    
    print(f"[done] Training complete. Progressive accuracy: {train_acc:.4f}")


if __name__ == "__main__":
    main()
