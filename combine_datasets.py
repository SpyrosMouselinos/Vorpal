#!/usr/bin/env python3
"""
Combine datasets from multiple LLM providers (Claude, Gemini, OpenAI) into unified train/valid sets.

Usage:
    python combine_datasets.py
"""

import csv
import random
from pathlib import Path
from typing import List, Tuple


# Input files from different providers
PROVIDER_FILES = {
    'claude': {
        'train': 'data/train_generated.csv',
        'valid': 'data/valid_generated.csv'
    },
    'gemini': {
        'train': 'data/train_gemini.csv',
        'valid': 'data/valid_gemini.csv'
    },
    'openai': {
        'train': 'data/train_openai.csv',
        'valid': 'data/valid_openai.csv'
    }
}

# Output files
OUTPUT_TRAIN = 'data/train_combined.csv'
OUTPUT_VALID = 'data/valid_combined.csv'
OUTPUT_STATS = 'data/combined_stats.txt'


def load_csv_data(filepath: str, provider: str) -> List[Tuple]:
    """Load data from CSV and tag with provider."""
    data = []
    if not Path(filepath).exists():
        print(f"  ⚠️  File not found: {filepath}")
        return data
    
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((
                int(row['category_id']),
                row['category_name'],
                int(row['subcategory_id']),
                row['subcategory_name'],
                row['caption'],
                provider  # Tag source
            ))
    
    return data


def combine_and_shuffle(
    data_dict: dict,
    seed: int = 42
) -> List[Tuple]:
    """Combine data from all providers and shuffle."""
    all_data = []
    
    for provider, data in data_dict.items():
        all_data.extend(data)
        print(f"  {provider}: {len(data):,} examples")
    
    # Shuffle to mix providers
    random.seed(seed)
    random.shuffle(all_data)
    
    return all_data


def save_combined_csv(
    data: List[Tuple],
    output_path: str,
    include_provider: bool = True
):
    """Save combined data to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['category_id', 'category_name', 'subcategory_id', 'subcategory_name', 'caption']
    if include_provider:
        fieldnames.append('source_provider')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    print(f"  Saved {len(data):,} examples to {output_path}")


def generate_statistics(
    train_data: List,
    valid_data: List,
    provider_stats: dict
) -> str:
    """Generate statistics report."""
    lines = []
    lines.append("="*70)
    lines.append("COMBINED DATASET STATISTICS")
    lines.append("="*70)
    
    lines.append("\nProvider Contributions (Training):")
    for provider, count in provider_stats['train'].items():
        pct = 100 * count / len(train_data) if train_data else 0
        lines.append(f"  {provider.capitalize():10} {count:6,} examples ({pct:5.1f}%)")
    
    lines.append("\nProvider Contributions (Validation):")
    for provider, count in provider_stats['valid'].items():
        pct = 100 * count / len(valid_data) if valid_data else 0
        lines.append(f"  {provider.capitalize():10} {count:6,} examples ({pct:5.1f}%)")
    
    lines.append(f"\nTotal Training Examples:   {len(train_data):,}")
    lines.append(f"Total Validation Examples: {len(valid_data):,}")
    lines.append(f"Total Examples:            {len(train_data) + len(valid_data):,}")
    
    lines.append("\n" + "="*70)
    
    return '\n'.join(lines)


def main():
    print("="*70)
    print("COMBINING DATASETS FROM MULTIPLE LLM PROVIDERS")
    print("="*70)
    
    # Load data from each provider
    print("\nLoading datasets from providers...")
    
    train_providers = {}
    valid_providers = {}
    
    for provider, files in PROVIDER_FILES.items():
        print(f"\n{provider.capitalize()}:")
        train_data = load_csv_data(files['train'], provider)
        valid_data = load_csv_data(files['valid'], provider)
        
        if train_data:
            train_providers[provider] = train_data
            print(f"  ✓ Loaded {len(train_data):,} training examples")
        if valid_data:
            valid_providers[provider] = valid_data
            print(f"  ✓ Loaded {len(valid_data):,} validation examples")
    
    if not train_providers:
        print("\n❌ Error: No training data found from any provider!")
        print("Please run at least one generation script first:")
        print("  python generate_data_with_claude.py")
        print("  python generate_data_with_gemini.py")
        print("  python generate_data_with_openai.py")
        return
    
    # Combine training data
    print("\n" + "-"*70)
    print("Combining training data...")
    combined_train = combine_and_shuffle(train_providers)
    
    # Combine validation data
    print("\nCombining validation data...")
    combined_valid = combine_and_shuffle(valid_providers, seed=43)
    
    # Save combined datasets
    print("\n" + "-"*70)
    print("Saving combined datasets...")
    save_combined_csv(combined_train, OUTPUT_TRAIN, include_provider=True)
    save_combined_csv(combined_valid, OUTPUT_VALID, include_provider=True)
    
    # Generate statistics
    provider_train_stats = {p: len(d) for p, d in train_providers.items()}
    provider_valid_stats = {p: len(d) for p, d in valid_providers.items()}
    
    stats = generate_statistics(
        combined_train,
        combined_valid,
        {'train': provider_train_stats, 'valid': provider_valid_stats}
    )
    
    print("\n" + stats)
    
    # Save statistics
    with open(OUTPUT_STATS, 'w') as f:
        f.write(stats)
    print(f"\nStatistics saved to: {OUTPUT_STATS}")
    
    print("\n✓ Dataset combination complete!")
    print(f"\nNext steps:")
    print(f"  1. Review combined data: {OUTPUT_TRAIN}")
    print(f"  2. Update configs to use combined datasets:")
    print(f"     configs/config_coarse.yaml: train_path: '{OUTPUT_TRAIN}'")
    print(f"     configs/config_fine.yaml: train_path: '{OUTPUT_TRAIN}'")
    print(f"  3. Train models on diverse, multi-provider data")


if __name__ == "__main__":
    main()

