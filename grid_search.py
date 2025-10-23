#!/usr/bin/env python3
"""
Grid search for optimal VW hyperparameters.

Usage:
    # Full grid search
    python grid_search.py --model coarse --parallel 4
    
    # Quick search (reduced grid)
    python grid_search.py --model coarse --quick
    
    # Random search
    python grid_search.py --model coarse --random 20
"""

import argparse
import csv
import itertools
import json
import random
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count
from datetime import datetime

import vowpal_wabbit_next as vw
from tqdm import tqdm


# Grid configurations
COARSE_GRID = {
    'bits': [24, 25, 26],
    'learning_rate': [0.1, 0.2, 0.3, 0.5],
    'ngram': [2, 3],
    'l2': [1e-7, 1e-6, 1e-5]
}

FINE_GRID = {
    'bits': [25, 26, 27],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'ngram': [2, 3],
    'l2': [1e-7, 1e-6, 1e-5]
}

QUICK_COARSE_GRID = {
    'bits': [24, 26],
    'learning_rate': [0.2, 0.5],
    'ngram': [2, 3],
    'l2': [1e-7, 1e-6]
}

QUICK_FINE_GRID = {
    'bits': [25, 27],
    'learning_rate': [0.1, 0.3],
    'ngram': [2, 3],
    'l2': [1e-7, 1e-6]
}


def clean_text(text: str) -> str:
    """Clean text for VW format."""
    return text.replace("|", " ").replace(":", " ").replace("\t", " ").strip()


def generate_ngrams(tokens, n=2):
    """Generate n-grams from tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams


def to_vw_line(label: int, text: str, namespace: str = "t", ngram: int = 2) -> str:
    """Convert to VW format with n-grams."""
    txt = clean_text(text)
    tokens = txt.lower().split()
    
    features = tokens.copy()
    if ngram >= 2 and len(tokens) >= 2:
        features.extend(generate_ngrams(tokens, 2))
    if ngram >= 3 and len(tokens) >= 3:
        features.extend(generate_ngrams(tokens, 3))
    
    feature_str = ' '.join(features)
    return f"{int(label)} |{namespace} {feature_str}"


def to_vw_line_unlabeled(text: str, namespace: str = "t", ngram: int = 2) -> str:
    """Convert to VW format without label."""
    txt = clean_text(text)
    tokens = txt.lower().split()
    
    features = tokens.copy()
    if ngram >= 2 and len(tokens) >= 2:
        features.extend(generate_ngrams(tokens, 2))
    if ngram >= 3 and len(tokens) >= 3:
        features.extend(generate_ngrams(tokens, 3))
    
    feature_str = ' '.join(features)
    return f"|{namespace} {feature_str}"


def load_data(filepath: str, label_column: str):
    """Load training or validation data from CSV."""
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row[label_column])
            caption = row['caption']
            data.append((label, caption))
    return data


def train_and_evaluate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Train a model with given hyperparameters and evaluate it."""
    config = params['config']
    model_type = params['model_type']
    train_data = params['train_data']
    valid_data = params['valid_data']
    
    # Build VW arguments
    vw_args = [
        "--oaa", str(config['num_classes']),
        "--loss_function", "logistic",
        "-b", str(config['bits']),
        "--learning_rate", str(config['learning_rate']),
        "--l2", str(config['l2']),
        "--probabilities"
    ]
    
    # Create workspace
    ws = vw.Workspace(vw_args)
    parser = vw.TextFormatParser(ws)
    
    # Training
    correct_train = 0
    for label, text in train_data:
        ex = parser.parse_line(to_vw_line(label, text, "t", config['ngram']))
        pred = ws.predict_then_learn_one(ex)
        
        if ws.prediction_type.name == "Scalars":
            yhat = 1 + max(range(len(pred)), key=lambda k: pred[k])
        else:
            yhat = int(pred)
        
        if yhat == label:
            correct_train += 1
    
    train_acc = correct_train / len(train_data)
    
    # Validation
    correct_valid = 0
    for label, text in valid_data:
        ex = parser.parse_line(to_vw_line_unlabeled(text, "t", config['ngram']))
        pred = ws.predict_one(ex)
        
        if ws.prediction_type.name == "Scalars":
            yhat = 1 + max(range(len(pred)), key=lambda k: pred[k])
        else:
            yhat = int(pred)
        
        if yhat == label:
            correct_valid += 1
    
    valid_acc = correct_valid / len(valid_data)
    
    # Don't return workspace (not picklable for multiprocessing)
    return {
        'model_type': model_type,
        'bits': config['bits'],
        'learning_rate': config['learning_rate'],
        'ngram': config['ngram'],
        'l2': config['l2'],
        'train_accuracy': train_acc,
        'valid_accuracy': valid_acc,
        'train_correct': correct_train,
        'valid_correct': correct_valid
    }


def generate_grid_combinations(grid: Dict[str, List], num_classes: int, model_type: str) -> List[Dict]:
    """Generate all combinations of hyperparameters."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    
    combinations = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config['num_classes'] = num_classes
        combinations.append({
            'config': config,
            'model_type': model_type
        })
    
    return combinations


def save_results(results: List[Dict], output_path: str):
    """Save all results to CSV."""
    if not results:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['model_type', 'bits', 'learning_rate', 'ngram', 'l2', 
                  'train_accuracy', 'valid_accuracy', 'train_correct', 'valid_correct']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Results saved to: {output_path}")


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    """Save intermediate results for resume capability."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """Load checkpoint if it exists."""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return []


def print_summary(results: List[Dict], model_type: str, top_k: int = 5):
    """Print summary of top configurations."""
    sorted_results = sorted(results, key=lambda x: x['valid_accuracy'], reverse=True)
    
    print("\n" + "="*80)
    print(f"GRID SEARCH RESULTS - {model_type.upper()} MODEL")
    print("="*80)
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Best validation accuracy: {sorted_results[0]['valid_accuracy']:.4f}")
    
    print(f"\nTop {top_k} Configurations:")
    print("-"*80)
    print(f"{'Rank':<6} {'Valid Acc':<12} {'Bits':<8} {'LR':<10} {'N-gram':<10} {'L2':<12}")
    print("-"*80)
    
    for i, result in enumerate(sorted_results[:top_k], 1):
        print(f"{i:<6} {result['valid_accuracy']:<12.4f} {result['bits']:<8} "
              f"{result['learning_rate']:<10.3f} {result['ngram']:<10} {result['l2']:<12.2e}")
    
    print("="*80)
    
    return sorted_results[0]


def main():
    parser = argparse.ArgumentParser(description="Grid search for VW hyperparameters")
    parser.add_argument("--model", required=True, choices=['coarse', 'fine'],
                       help="Which model to optimize")
    parser.add_argument("--parallel", type=int, default=cpu_count(),
                       help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--quick", action="store_true",
                       help="Use reduced grid for faster search")
    parser.add_argument("--random", type=int,
                       help="Random search with N samples instead of full grid")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"GRID SEARCH HYPERPARAMETER OPTIMIZATION - {args.model.upper()} MODEL")
    print("="*80)
    
    # Set up paths and parameters
    if args.model == 'coarse':
        num_classes = 45
        label_column = 'category_id'
        grid = QUICK_COARSE_GRID if args.quick else COARSE_GRID
        results_file = 'results/grid_search_coarse.csv'
        checkpoint_file = 'results/grid_search_coarse_checkpoint.json'
        best_model_path = 'models/model_coarse_best.vwbin'
        best_config_path = 'configs/config_coarse_best.yaml'
    else:
        num_classes = 422
        label_column = 'subcategory_id'
        grid = QUICK_FINE_GRID if args.quick else FINE_GRID
        results_file = 'results/grid_search_fine.csv'
        checkpoint_file = 'results/grid_search_fine_checkpoint.json'
        best_model_path = 'models/model_fine_best.vwbin'
        best_config_path = 'configs/config_fine_best.yaml'
    
    # Load data
    print("\nLoading data...")
    train_data = load_data('data/train_combined.csv', label_column)
    valid_data = load_data('data/valid_combined.csv', label_column)
    print(f"  Training: {len(train_data):,} examples")
    print(f"  Validation: {len(valid_data):,} examples")
    
    # Generate combinations
    print("\nGenerating hyperparameter combinations...")
    combinations = generate_grid_combinations(grid, num_classes, args.model)
    
    if args.random:
        print(f"  Random sampling {args.random} configurations from {len(combinations)} total")
        combinations = random.sample(combinations, min(args.random, len(combinations)))
    
    print(f"  Total configurations to test: {len(combinations)}")
    
    # Check for existing results
    completed_configs = set()
    results = []
    
    if args.resume and Path(checkpoint_file).exists():
        print(f"\nResuming from checkpoint: {checkpoint_file}")
        checkpoint_results = load_checkpoint(checkpoint_file)
        results = checkpoint_results
        
        # Track completed configurations
        for r in checkpoint_results:
            config_key = (r['bits'], r['learning_rate'], r['ngram'], r['l2'])
            completed_configs.add(config_key)
        
        print(f"  Loaded {len(results)} previous results")
    
    # Filter out completed configurations
    remaining_combinations = []
    for combo in combinations:
        config_key = (combo['config']['bits'], combo['config']['learning_rate'],
                     combo['config']['ngram'], combo['config']['l2'])
        if config_key not in completed_configs:
            combo['train_data'] = train_data
            combo['valid_data'] = valid_data
            remaining_combinations.append(combo)
    
    print(f"  Remaining configurations to test: {len(remaining_combinations)}")
    
    if not remaining_combinations:
        print("\n✓ All configurations already tested!")
        if results:
            best = print_summary(results, args.model)
            return
        else:
            print("No results found. Remove checkpoint to start fresh.")
            return
    
    # Estimate time
    est_time_per_config = 5 if args.model == 'coarse' else 15  # seconds
    est_total_minutes = (len(remaining_combinations) * est_time_per_config) / (60 * args.parallel)
    
    print(f"\nEstimated time: ~{est_total_minutes:.0f} minutes with {args.parallel} workers")
    print(f"Grid parameters:")
    for param, values in grid.items():
        print(f"  {param}: {values}")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Run grid search with multiprocessing
    print(f"\nRunning grid search with {args.parallel} parallel workers...")
    
    with Pool(processes=args.parallel) as pool:
        # Use imap for progress tracking
        new_results = list(tqdm(
            pool.imap(train_and_evaluate, remaining_combinations),
            total=len(remaining_combinations),
            desc="Grid search progress"
        ))
    
    # Combine with previous results
    results.extend(new_results)
    
    # Save final results
    save_results(results, results_file)
    save_checkpoint(results, checkpoint_file)
    
    # Find and save best model
    best = print_summary(results, args.model)
    
    # Save best model configuration
    print(f"\nSaving best configuration...")
    
    # The workspace is not serializable after multiprocessing, so retrain best model
    print("  Retraining best model to save...")
    best_params = {
        'config': {
            'num_classes': num_classes,
            'bits': best['bits'],
            'learning_rate': best['learning_rate'],
            'ngram': best['ngram'],
            'l2': best['l2']
        },
        'model_type': args.model,
        'train_data': train_data,
        'valid_data': valid_data
    }
    
    # Train best model with workspace saving
    vw_args = [
        "--oaa", str(num_classes),
        "--loss_function", "logistic",
        "-b", str(best['bits']),
        "--learning_rate", str(best['learning_rate']),
        "--l2", str(best['l2']),
        "--probabilities"
    ]
    
    ws = vw.Workspace(vw_args)
    parser = vw.TextFormatParser(ws)
    
    # Train on full training set
    for label, text in train_data:
        ex = parser.parse_line(to_vw_line(label, text, "t", best['ngram']))
        ws.learn_one(ex)
    
    # Save best model
    with open(best_model_path, 'wb') as f:
        f.write(ws.serialize())
    print(f"  ✓ Best model saved to: {best_model_path}")
    
    # Save best config
    best_config = {
        'model': {
            'num_classes': num_classes,
            'namespace': 't'
        },
        'vw': {
            'bits': best['bits'],
            'learning_rate': best['learning_rate'],
            'l2': best['l2'],
            'ngram': best['ngram'],
            'loss_function': 'logistic',
            'probabilities': True
        },
        'data': {
            'train_path': 'data/train_combined.csv',
            'valid_path': 'data/valid_combined.csv',
            'text_column': 'caption',
            'label_column': label_column
        },
        'training': {
            'log_every': 5000,
            'checkpoint_path': best_model_path,
            'save_labels': False
        },
        'inference': {
            'confidence_threshold': 0.85 if args.model == 'coarse' else 0.70
        }
    }
    
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ Best config saved to: {best_config_path}")
    
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"\nBest configuration achieves {best['valid_accuracy']:.4f} validation accuracy")
    print(f"\nUse the optimized model:")
    print(f"  python eval.py --config {best_config_path} --checkpoint {best_model_path}")


if __name__ == "__main__":
    main()

