#!/usr/bin/env python3
"""
inference_vw.py - Run inference with a trained VW Next model.

Usage examples:
    # From stdin
    echo "football match highlights" | python inference_vw.py --config config.example.yaml --checkpoint model.vw --stdin
    
    # From CSV
    python inference_vw.py --config config.example.yaml --checkpoint model.vw --input_csv unlabeled.csv --text_column caption
    
    # Interactive annotation (auto-label if confident, else ask)
    echo "ambiguous caption" | python inference_vw.py --config config.example.yaml --checkpoint model.vw --stdin --interactive --conf_thresh 0.85
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
import vowpal_wabbit_next as vw


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_label_names(checkpoint_path: str) -> Optional[Dict[int, str]]:
    """Try to load label names from .labels.json next to checkpoint."""
    labels_path = checkpoint_path.replace('.vw', '.labels.json').replace('.vwbin', '.labels.json')
    if Path(labels_path).exists():
        with open(labels_path, 'r') as f:
            mapping = json.load(f)  # str -> int
            # Reverse to int -> str
            return {v: k for k, v in mapping.items()}
    return None


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


def predict_one(
    ws: vw.Workspace,
    parser: vw.TextFormatParser,
    text: str,
    namespace: str,
    ngram: int = 2
) -> tuple:
    """
    Predict class and confidence for a single text.
    
    Returns:
        (predicted_class: int, confidence: float, all_probs: List[float])
    """
    ex = parser.parse_line(to_vw_line(None, text, namespace, ngram))
    pred = ws.predict_one(ex)
    
    pred_type = ws.prediction_type
    
    if pred_type.name == "Scalars":
        probs = list(pred)
        k_hat = 1 + max(range(len(probs)), key=lambda k: probs[k])
        p_hat = probs[k_hat - 1]
        return k_hat, p_hat, probs
    elif pred_type.name == "Multiclass":
        k_hat = int(pred)
        return k_hat, 1.0, []
    else:
        raise RuntimeError(f"Unexpected prediction type: {pred_type}")


def classify_or_query(
    ws: vw.Workspace,
    parser: vw.TextFormatParser,
    text: str,
    namespace: str,
    ngram: int,
    conf_thresh: float,
    interactive: bool = False,
    label_names: Optional[Dict[int, str]] = None
) -> tuple:
    """
    Classify text; if confidence < threshold and interactive=True, ask user for label.
    
    Returns:
        (final_label: int, confidence: float, was_queried: bool)
    """
    k_hat, p_hat, probs = predict_one(ws, parser, text, namespace, ngram)
    
    if p_hat >= conf_thresh or not interactive:
        # Auto-accept
        if interactive:
            # Learn with pseudo-label
            ex = parser.parse_line(to_vw_line(k_hat, text, namespace, ngram))
            ws.learn_one(ex)
        return k_hat, p_hat, False
    else:
        # Query user
        print(f"\n[QUERY] Low confidence ({p_hat:.3f}) for text:", file=sys.stderr)
        print(f"  \"{text[:100]}...\"", file=sys.stderr)
        
        label_str = label_names.get(k_hat, str(k_hat)) if label_names else str(k_hat)
        print(f"  Predicted: {label_str} (class {k_hat})", file=sys.stderr)
        print(f"  Enter correct class ID (1..{len(probs)} or press Enter to accept): ", end='', file=sys.stderr)
        
        user_input = input().strip()
        if user_input:
            y = int(user_input)
        else:
            y = k_hat
        
        # Learn with true label
        ex = parser.parse_line(to_vw_line(y, text, namespace, ngram))
        ws.learn_one(ex)
        
        return y, p_hat, True


def main():
    parser = argparse.ArgumentParser(description="Inference with VW Next model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    
    # Input sources
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--input_csv", help="Path to CSV with unlabeled data")
    parser.add_argument("--text_column", help="Column name for text (required if --input_csv)")
    
    # Interactive annotation
    parser.add_argument("--interactive", action="store_true", help="Enable interactive annotation")
    parser.add_argument("--conf_thresh", type=float, help="Confidence threshold for auto-labeling")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    print(f"[load] Loading model from: {args.checkpoint}", file=sys.stderr)
    with open(args.checkpoint, 'rb') as f:
        model_data = f.read()
    
    ws = vw.Workspace(model_data=model_data)
    text_parser = vw.TextFormatParser(ws)
    
    # Load label names if available
    label_names = load_label_names(args.checkpoint)
    if label_names:
        print(f"[load] Loaded {len(label_names)} label names", file=sys.stderr)
    
    # Get namespace and ngram setting
    namespace = cfg['model']['namespace']
    ngram = cfg.get('vw', {}).get('ngram', 2)
    
    # Get confidence threshold
    conf_thresh = args.conf_thresh or cfg['inference'].get('confidence_threshold', 0.85)
    
    # Process input
    if args.stdin:
        # Read from stdin
        print("[inference] Reading from stdin...", file=sys.stderr)
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            
            if args.interactive:
                label, conf, queried = classify_or_query(
                    ws, text_parser, text, namespace, ngram, conf_thresh, True, label_names
                )
                label_str = label_names.get(label, str(label)) if label_names else str(label)
                status = "QUERIED" if queried else "AUTO"
                print(f"{label}\t{conf:.4f}\t{status}\t{label_str}\t{text}")
            else:
                label, conf, _ = predict_one(ws, text_parser, text, namespace, ngram)
                label_str = label_names.get(label, str(label)) if label_names else str(label)
                print(f"{label}\t{conf:.4f}\t{label_str}\t{text}")
    
    elif args.input_csv:
        # Read from CSV
        if not args.text_column:
            parser.error("--text_column required when using --input_csv")
        
        print(f"[inference] Reading from CSV: {args.input_csv}", file=sys.stderr)
        import csv
        
        with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Print header
            print("predicted_label\tconfidence\tlabel_name\ttext")
            
            for i, row in enumerate(reader, 1):
                text = row[args.text_column]
                
                if args.interactive:
                    label, conf, queried = classify_or_query(
                        ws, text_parser, text, namespace, ngram, conf_thresh, True, label_names
                    )
                    label_str = label_names.get(label, str(label)) if label_names else str(label)
                    status = "QUERIED" if queried else "AUTO"
                    print(f"{label}\t{conf:.4f}\t{status}\t{label_str}\t{text}")
                else:
                    label, conf, _ = predict_one(ws, text_parser, text, namespace, ngram)
                    label_str = label_names.get(label, str(label)) if label_names else str(label)
                    print(f"{label}\t{conf:.4f}\t{label_str}\t{text}")
                
                if i % 10000 == 0:
                    print(f"[progress] Processed {i:,} rows", file=sys.stderr)
        
        print(f"[done] Processed {i:,} total rows", file=sys.stderr)
    
    else:
        parser.error("Must specify either --stdin or --input_csv")


if __name__ == "__main__":
    main()
