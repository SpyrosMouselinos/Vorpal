#!/usr/bin/env python3
"""
eval_vw.py - Evaluate a saved VW Next model on validation data.

Usage:
    python eval_vw.py --config config.example.yaml --checkpoint model.vw
"""

import argparse
from typing import Iterator, Tuple, Optional, Dict, Any

import yaml
import vowpal_wabbit_next as vw


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Clean text for VW format."""
    return text.replace("|", " ").replace(":", " ").replace("\t", " ").strip()


def generate_ngrams(tokens, n=2):
    """Generate n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams


def to_vw_line(label: Optional[int], text: str, namespace: str = "t", ngram: int = 2) -> str:
    """Convert to VW format line with n-grams."""
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
    """Yield (label_id, text) from CSV or Parquet."""
    from pathlib import Path
    path_obj = Path(path)
    
    if path_obj.suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("Install pyarrow: pip install pyarrow")
        
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                label_raw = row[label_col]
                text = str(row[text_col])
                
                if label_mapping and isinstance(label_raw, str):
                    label_id = label_mapping[label_raw]
                else:
                    label_id = int(label_raw)
                
                yield label_id, text
    
    elif path_obj.suffix == '.csv':
        import csv
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_raw = row[label_col]
                text = str(row[text_col])
                
                if label_mapping and isinstance(label_raw, str):
                    label_id = label_mapping[label_raw]
                else:
                    label_id = int(label_raw)
                
                yield label_id, text
    else:
        raise ValueError(f"Unsupported format: {path_obj.suffix}")


def evaluate(
    ws: vw.Workspace,
    parser: vw.TextFormatParser,
    data_iter: Iterator[Tuple[int, str]],
    namespace: str,
    ngram: int,
    log_every: int = 100000
) -> Dict[str, float]:
    """
    Evaluate model on held-out data (no learning).
    
    Returns:
        Dict with metrics (accuracy, total, correct)
    """
    pred_type = ws.prediction_type
    seen = correct = 0
    
    all_probs = []  # for computing mean confidence
    
    for i, (y, text) in enumerate(data_iter, 1):
        ex = parser.parse_line(to_vw_line(None, text, namespace, ngram))  # unlabeled
        pred = ws.predict_one(ex)  # no learning
        
        if pred_type.name == "Scalars":
            probs = list(pred)
            yhat = 1 + max(range(len(probs)), key=lambda k: probs[k])
            max_prob = max(probs)
            all_probs.append(max_prob)
        elif pred_type.name == "Multiclass":
            yhat = int(pred)
            max_prob = None
        else:
            raise RuntimeError(f"Unexpected prediction type: {pred_type}")
        
        correct += (yhat == y)
        seen += 1
        
        if i % log_every == 0:
            print(f"[eval] examples={seen:,} accuracy={correct/seen:.4f}")
    
    accuracy = correct / max(seen, 1)
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": seen,
    }
    
    if all_probs:
        metrics["mean_confidence"] = sum(all_probs) / len(all_probs)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VW Next model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    print(f"[load] Loading model from: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        model_data = f.read()
    
    ws = vw.Workspace(model_data=model_data)
    text_parser = vw.TextFormatParser(ws)
    
    # Evaluate on validation set
    print(f"[eval] Loading validation data from: {cfg['data']['valid_path']}")
    valid_iter = iter_data(
        cfg['data']['valid_path'],
        cfg['data']['text_column'],
        cfg['data']['label_column'],
        cfg['data'].get('label_mapping')
    )
    
    metrics = evaluate(
        ws,
        text_parser,
        valid_iter,
        cfg['model']['namespace'],
        cfg.get('vw', {}).get('ngram', 2)
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Correct:         {metrics['correct']:,}")
    print(f"Total:           {metrics['total']:,}")
    if 'mean_confidence' in metrics:
        print(f"Mean confidence: {metrics['mean_confidence']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
