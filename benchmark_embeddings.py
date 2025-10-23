#!/usr/bin/env python3
"""
Benchmark different embedding models and classifiers systematically.

Usage:
    # Test all embeddings (Phase 1)
    python benchmark_embeddings.py --phase 1
    
    # Test classifiers with best embedding (Phase 2)
    python benchmark_embeddings.py --phase 2
    
    # Test ensembles (Phase 3)
    python benchmark_embeddings.py --phase 3
    
    # Run all phases
    python benchmark_embeddings.py --all
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np

from models.embedding_classifier import EmbeddingClassifier
from utils.embedding_cache import load_embeddings_cache, save_embeddings_cache


# Embedding models to test
BASE_MODELS = [
    'BAAI/bge-base-en-v1.5',      # State-of-the-art
    'intfloat/e5-base-v2',        # Microsoft's best
    'thenlper/gte-base',          # Alibaba
    'all-mpnet-base-v2',          # Current baseline
]

SMALL_MODELS = [
    'BAAI/bge-small-en-v1.5',     # Best small
    'intfloat/e5-small-v2',       # Microsoft small
    'all-MiniLM-L6-v2',           # Current small baseline
]

LARGE_MODELS = [
    'BAAI/bge-large-en-v1.5',     # Best overall
    'intfloat/e5-large-v2',       # Microsoft large
]

ALL_EMBEDDING_MODELS = BASE_MODELS + SMALL_MODELS + LARGE_MODELS

# Classifiers to test
CLASSIFIERS = [
    'logistic_regression',
    'xgboost',
    'lightgbm',
    'svm'
]


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


def benchmark_embedding(
    encoder_name: str,
    train_texts: List[str],
    train_labels: List[int],
    valid_texts: List[str],
    valid_labels: List[int],
    dataset_name: str = "train_combined",
    classifier_type: str = 'logistic_regression'
) -> dict:
    """Benchmark a single embedding model."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {encoder_name}")
    print(f"{'='*70}")
    
    # Initialize classifier
    classifier = EmbeddingClassifier(
        encoder_name=encoder_name,
        classifier_type=classifier_type
    )
    
    # Check cache
    cached = load_embeddings_cache(dataset_name, encoder_name)
    
    if cached is not None:
        embeddings_train, _ = cached
        print(f"Using cached training embeddings")
        
        # Train classifier directly
        start_time = time.time()
        classifier._init_encoder()
        classifier._init_classifier(len(set(train_labels)))
        classifier.classifier.fit(embeddings_train, train_labels)
        train_time = time.time() - start_time
        
        encode_time = 0  # Already cached
    else:
        # Encode and cache
        start_time = time.time()
        embeddings_train = classifier.encode_texts(train_texts, batch_size=32)
        encode_time = time.time() - start_time
        
        # Cache for future runs
        save_embeddings_cache(embeddings_train, dataset_name, encoder_name)
        
        # Train classifier
        start_time = time.time()
        classifier._init_classifier(len(set(train_labels)))
        classifier.classifier.fit(embeddings_train, train_labels)
        train_time = time.time() - start_time
    
    # Evaluate
    print(f"Evaluating on validation set...")
    start_time = time.time()
    predictions = classifier.predict(valid_texts, batch_size=32)
    inference_time = (time.time() - start_time) / len(valid_texts) * 1000  # ms per example
    
    accuracy = np.mean(predictions == valid_labels)
    
    result = {
        'encoder': encoder_name,
        'classifier': classifier_type,
        'accuracy': accuracy,
        'correct': int(np.sum(predictions == valid_labels)),
        'total': len(valid_labels),
        'encode_time_sec': encode_time,
        'train_time_sec': train_time,
        'inference_ms_per_example': inference_time,
        'embedding_dim': embeddings_train.shape[1] if cached is None else cached[0].shape[1]
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")
    print(f"  Encoding time: {encode_time:.1f}s (cached: {cached is not None})")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Inference: {inference_time:.2f}ms per example")
    
    return result


def phase1_test_embeddings(model_type: str, models_to_test: List[str] = None):
    """Phase 1: Test all embedding models with LogisticRegression."""
    
    print("\n" + "="*70)
    print("PHASE 1: TESTING EMBEDDING MODELS")
    print("="*70)
    
    # Determine label column and dataset
    if model_type == 'coarse':
        label_column = 'category_id'
        output_file = 'results/embedding_benchmark_coarse.csv'
    else:
        label_column = 'subcategory_id'
        output_file = 'results/embedding_benchmark_fine.csv'
    
    # Load data
    print(f"\nLoading data...")
    train_texts, train_labels = load_data('data/train_combined.csv', label_column)
    valid_texts, valid_labels = load_data('data/valid_combined.csv', label_column)
    
    print(f"  Training: {len(train_texts):,} examples")
    print(f"  Validation: {len(valid_texts):,} examples")
    print(f"  Classes: {len(set(train_labels))}")
    
    # Test models
    if models_to_test is None:
        models_to_test = ALL_EMBEDDING_MODELS
    
    print(f"\nTesting {len(models_to_test)} embedding models...")
    
    results = []
    for i, encoder_name in enumerate(models_to_test, 1):
        print(f"\n[{i}/{len(models_to_test)}] {encoder_name}")
        
        try:
            result = benchmark_embedding(
                encoder_name,
                train_texts,
                train_labels,
                valid_texts,
                valid_labels,
                dataset_name='train_combined',
                classifier_type='logistic_regression'
            )
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE - Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Show summary
    if results:
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nTop 5 Embedding Models ({model_type}):")
        print("-"*70)
        print(f"{'Rank':<6} {'Encoder':<40} {'Accuracy':<12} {'Inference'}")
        print("-"*70)
        
        for i, r in enumerate(sorted_results[:5], 1):
            encoder_short = r['encoder'].split('/')[-1][:38]
            print(f"{i:<6} {encoder_short:<40} {r['accuracy']:<12.4f} {r['inference_ms_per_example']:.2f}ms")
        
        print("-"*70)
        print(f"\n✓ Best model: {sorted_results[0]['encoder']}")
        print(f"  Accuracy: {sorted_results[0]['accuracy']:.4f}")
        print(f"  Inference: {sorted_results[0]['inference_ms_per_example']:.2f}ms")
        
        return sorted_results[0]['encoder']
    
    return None


def phase2_test_classifiers(model_type: str, best_encoder: str):
    """Phase 2: Test classifiers with best embedding."""
    
    print("\n" + "="*70)
    print(f"PHASE 2: TESTING CLASSIFIERS WITH {best_encoder}")
    print("="*70)
    
    # Determine label column
    if model_type == 'coarse':
        label_column = 'category_id'
        output_file = 'results/classifier_benchmark_coarse.csv'
    else:
        label_column = 'subcategory_id'
        output_file = 'results/classifier_benchmark_fine.csv'
    
    # Load data
    print(f"\nLoading data...")
    train_texts, train_labels = load_data('data/train_combined.csv', label_column)
    valid_texts, valid_labels = load_data('data/valid_combined.csv', label_column)
    
    print(f"  Training: {len(train_texts):,} examples")
    print(f"  Validation: {len(valid_texts):,} examples")
    
    # Test classifiers
    print(f"\nTesting {len(CLASSIFIERS)} classifiers...")
    
    results = []
    for i, classifier_type in enumerate(CLASSIFIERS, 1):
        print(f"\n[{i}/{len(CLASSIFIERS)}] {classifier_type}")
        
        try:
            result = benchmark_embedding(
                best_encoder,
                train_texts,
                train_labels,
                valid_texts,
                valid_labels,
                dataset_name='train_combined',
                classifier_type=classifier_type
            )
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE - Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Show summary
    if results:
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nClassifier Rankings ({model_type}):")
        print("-"*70)
        print(f"{'Rank':<6} {'Classifier':<25} {'Accuracy':<12} {'Train Time':<12}")
        print("-"*70)
        
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:<6} {r['classifier']:<25} {r['accuracy']:<12.4f} {r['train_time_sec']:.2f}s")
        
        print("-"*70)
        print(f"\n✓ Best classifier: {sorted_results[0]['classifier']}")
        print(f"  Accuracy: {sorted_results[0]['accuracy']:.4f}")
        
        return sorted_results[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models and classifiers")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="Which phase to run (1=embeddings, 2=classifiers, 3=ensembles)")
    parser.add_argument("--model", default='coarse', choices=['coarse', 'fine'],
                       help="Which model to benchmark")
    parser.add_argument("--all", action="store_true",
                       help="Run all phases sequentially")
    parser.add_argument("--quick", action="store_true",
                       help="Test only base models (faster)")
    
    args = parser.parse_args()
    
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           SYSTEMATIC MODEL BENCHMARKING SYSTEM                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    # Determine which models to test
    if args.quick:
        models_to_test = BASE_MODELS
        print(f"\nQuick mode: Testing {len(models_to_test)} base models only")
    else:
        models_to_test = ALL_EMBEDDING_MODELS
        print(f"\nFull mode: Testing {len(models_to_test)} embedding models")
    
    best_encoder = None
    best_config = None
    
    # Phase 1: Test embeddings
    if args.phase == 1 or args.all or args.phase is None:
        best_encoder = phase1_test_embeddings(args.model, models_to_test)
    
    # Phase 2: Test classifiers
    if args.phase == 2 or args.all:
        if best_encoder is None:
            # Load from Phase 1 results
            result_file = f'results/embedding_benchmark_{args.model}.csv'
            if Path(result_file).exists():
                with open(result_file, 'r') as f:
                    reader = csv.DictReader(f)
                    results = list(reader)
                    if results:
                        best_encoder = max(results, key=lambda x: float(x['accuracy']))['encoder']
                        print(f"\nLoaded best encoder from Phase 1: {best_encoder}")
            else:
                print(f"\n❌ Error: Run Phase 1 first to find best encoder")
                return
        
        best_config = phase2_test_classifiers(args.model, best_encoder)
    
    # Phase 3: Ensembles
    if args.phase == 3 or args.all:
        print("\n" + "="*70)
        print("PHASE 3: ENSEMBLE METHODS")
        print("="*70)
        print("\nEnsemble methods will be implemented after Phases 1 & 2 complete.")
        print("For now, use the best single model from Phase 2.")
    
    print("\n" + "="*70)
    print("BENCHMARKING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

