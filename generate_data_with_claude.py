#!/usr/bin/env python3
"""
Generate realistic video captions using Claude API for hierarchical classification.

Requirements:
    pip install anthropic tqdm

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python generate_data_with_claude.py
"""

import json
import csv
import os
import random
import time
from typing import Dict, List, Tuple
from pathlib import Path

try:
    import anthropic
    from tqdm import tqdm
except ImportError:
    print("Error: Missing dependencies. Please install:")
    print("  pip install anthropic tqdm")
    exit(1)


# Configuration
CAPTIONS_PER_SUBCATEGORY = 20
EXTRA_EVAL_SAMPLES = 1000
OUTPUT_TRAIN = "data/train_generated.csv"
OUTPUT_VALID = "data/valid_generated.csv"
OUTPUT_RAW_RESPONSES = "data/generation_log.jsonl"  # Save raw API responses
TAXONOMY_PATH = "data/taxonomy.json"
API_RETRY_DELAY = 2  # seconds between retries
MAX_RETRIES = 3


def load_taxonomy(path: str = TAXONOMY_PATH) -> Dict:
    """Load the taxonomy from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_keywords_to_avoid(category_name: str, subcategory_name: str) -> List[str]:
    """
    Extract keywords that should be avoided in the generated captions.
    This prevents the model from just memorizing category names.
    """
    # Combine category and subcategory text
    combined = f"{category_name} {subcategory_name}".lower()
    
    # Remove common words and punctuation
    stop_words = {'&', 'and', 'the', 'a', 'an', 'of', 'in', 'at', 'to', 'for', 'with', 'on'}
    words = combined.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace('-', ' ').split()
    
    # Keep only meaningful words (3+ chars, not stop words)
    keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
    
    return keywords


def generate_caption_prompt(
    category_name: str,
    subcategory_name: str,
    keywords_to_avoid: List[str],
    num_captions: int = 20
) -> str:
    """
    Create a prompt for Claude to generate realistic video captions.
    """
    keywords_str = ", ".join(keywords_to_avoid)
    
    prompt = f"""You are generating realistic video descriptions for a video classification dataset.

Category: {category_name}
Subcategory: {subcategory_name}

Generate {num_captions} diverse, realistic video descriptions that would fit this subcategory. 

CRITICAL REQUIREMENTS:
1. Each description should be 10-25 words long
2. Describe what is VISUALLY happening in the video (action, setting, subjects, mood)
3. Make them sound like actual video metadata/captions
4. DO NOT use these exact words: {keywords_str}
5. Instead, describe the scene, actions, and visual elements indirectly
6. Be creative and varied - avoid repetitive phrases
7. Include specific visual details (colors, movements, environments, emotions)
8. Mix different perspectives, times of day, settings, and subjects

Format: Return ONLY a JSON array of strings, nothing else. Example:
["description one here", "description two here", ...]

Generate {num_captions} unique descriptions now:"""
    
    return prompt


def call_claude_api(
    client: anthropic.Anthropic,
    prompt: str,
    max_tokens: int = 2000
) -> str:
    """Call Claude API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",  # Claude Haiku 4.5 - fastest and most cost-efficient
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(API_RETRY_DELAY * (attempt + 1))
            else:
                raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")


def parse_captions_from_response(response: str) -> List[str]:
    """Parse the JSON array of captions from Claude's response."""
    try:
        # Try to parse as JSON
        captions = json.loads(response)
        if isinstance(captions, list):
            return [str(c).strip() for c in captions if c]
    except json.JSONDecodeError:
        # Fallback: try to extract from markdown code block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            captions = json.loads(json_str)
            return [str(c).strip() for c in captions if c]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            captions = json.loads(json_str)
            return [str(c).strip() for c in captions if c]
    
    # If all else fails, try line-by-line parsing
    lines = [l.strip(' "[],-') for l in response.split('\n') if l.strip()]
    return [l for l in lines if len(l) > 10 and not l.startswith('{')]


def generate_captions_for_subcategory(
    client: anthropic.Anthropic,
    category_name: str,
    subcategory_name: str,
    num_captions: int = CAPTIONS_PER_SUBCATEGORY
) -> List[str]:
    """Generate captions for a single subcategory."""
    keywords_to_avoid = extract_keywords_to_avoid(category_name, subcategory_name)
    
    prompt = generate_caption_prompt(
        category_name,
        subcategory_name,
        keywords_to_avoid,
        num_captions
    )
    
    response = call_claude_api(client, prompt)
    captions = parse_captions_from_response(response)
    
    # Ensure we have the right number
    if len(captions) < num_captions:
        print(f"  Warning: Only got {len(captions)}/{num_captions} captions, retrying...")
        # Retry once
        response = call_claude_api(client, prompt)
        additional = parse_captions_from_response(response)
        captions.extend(additional)
    
    return captions[:num_captions]


def generate_all_captions(
    api_key: str,
    taxonomy: Dict,
    save_responses: bool = True
) -> List[Tuple[int, str, int, str, str]]:
    """
    Generate all training captions.
    
    Returns:
        List of (category_id, category_name, subcategory_id, subcategory_name, caption)
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    all_data = []
    total_subcats = sum(len(cat['subcategories']) for cat in taxonomy.values())
    
    print(f"\nGenerating captions for {total_subcats} subcategories...")
    print(f"Target: {CAPTIONS_PER_SUBCATEGORY} captions per subcategory")
    print(f"Expected total: {total_subcats * CAPTIONS_PER_SUBCATEGORY} examples\n")
    
    # Open log file for raw responses
    log_file = None
    if save_responses:
        log_file = open(OUTPUT_RAW_RESPONSES, 'w', encoding='utf-8')
        print(f"Logging API responses to: {OUTPUT_RAW_RESPONSES}\n")
    
    try:
        with tqdm(total=total_subcats, desc="Overall progress") as pbar:
            for cat_id_str, cat_data in taxonomy.items():
                category_id = int(cat_id_str)
                category_name = cat_data['name']
                
                for subcat_id_str, subcat_name in cat_data['subcategories'].items():
                    subcategory_id = int(subcat_id_str)
                    
                    pbar.set_description(f"Cat {category_id}: {category_name[:30]}")
                    
                    try:
                        keywords_to_avoid = extract_keywords_to_avoid(category_name, subcat_name)
                        prompt = generate_caption_prompt(
                            category_name,
                            subcat_name,
                            keywords_to_avoid,
                            CAPTIONS_PER_SUBCATEGORY
                        )
                        
                        response = call_claude_api(client, prompt)
                        
                        # Log the raw response
                        if log_file:
                            log_entry = {
                                'category_id': category_id,
                                'category_name': category_name,
                                'subcategory_id': subcategory_id,
                                'subcategory_name': subcat_name,
                                'keywords_avoided': keywords_to_avoid,
                                'response': response
                            }
                            log_file.write(json.dumps(log_entry) + '\n')
                            log_file.flush()
                        
                        captions = parse_captions_from_response(response)
                        
                        # Ensure we have enough captions
                        if len(captions) < CAPTIONS_PER_SUBCATEGORY:
                            print(f"\n  Warning: Only got {len(captions)}/{CAPTIONS_PER_SUBCATEGORY} captions")
                        
                        for caption in captions[:CAPTIONS_PER_SUBCATEGORY]:
                            all_data.append((
                                category_id,
                                category_name,
                                subcategory_id,
                                subcat_name,
                                caption
                            ))
                        
                        # Small delay to avoid rate limits
                        time.sleep(0.3)
                        
                    except Exception as e:
                        print(f"\n  Error generating captions for {subcat_name}: {e}")
                        print(f"  Skipping this subcategory...")
                    
                    pbar.update(1)
    finally:
        if log_file:
            log_file.close()
    
    return all_data


def split_train_valid(
    data: List[Tuple[int, str, int, str, str]],
    num_valid: int = EXTRA_EVAL_SAMPLES
) -> Tuple[List, List]:
    """
    Split data into training and validation sets.
    """
    # Shuffle the data
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Split
    valid_data = shuffled[:num_valid]
    train_data = shuffled[num_valid:]
    
    return train_data, valid_data


def save_to_csv(
    data: List[Tuple[int, str, int, str, str]],
    output_path: str
):
    """Save data to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category_id', 'category_name', 'subcategory_id', 'subcategory_name', 'caption'])
        writer.writerows(data)
    
    print(f"  Saved {len(data)} examples to {output_path}")


def print_statistics(train_data: List, valid_data: List, taxonomy: Dict):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    total_cats = len(taxonomy)
    total_subcats = sum(len(cat['subcategories']) for cat in taxonomy.values())
    
    print(f"\nTaxonomy:")
    print(f"  Categories: {total_cats}")
    print(f"  Subcategories: {total_subcats}")
    
    print(f"\nGenerated Data:")
    print(f"  Training examples: {len(train_data):,}")
    print(f"  Validation examples: {len(valid_data):,}")
    print(f"  Total examples: {len(train_data) + len(valid_data):,}")
    
    print(f"\nPer Subcategory:")
    expected_per_subcat = CAPTIONS_PER_SUBCATEGORY
    actual_per_subcat = (len(train_data) + len(valid_data)) / total_subcats
    print(f"  Expected: {expected_per_subcat}")
    print(f"  Actual (avg): {actual_per_subcat:.1f}")
    
    # Sample captions
    print(f"\nSample Captions (from validation set):")
    for i, (cat_id, cat_name, sub_id, sub_name, caption) in enumerate(valid_data[:5]):
        print(f"\n  {i+1}. Category: {cat_name}")
        print(f"     Subcategory: {sub_name}")
        print(f"     Caption: {caption}")
    
    print("\n" + "="*70)


def main():
    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    print("="*70)
    print("GENERATING REALISTIC VIDEO CAPTIONS WITH CLAUDE")
    print("="*70)
    
    # Load taxonomy
    print(f"\nLoading taxonomy from {TAXONOMY_PATH}...")
    taxonomy = load_taxonomy()
    
    total_subcats = sum(len(cat['subcategories']) for cat in taxonomy.values())
    expected_total = total_subcats * CAPTIONS_PER_SUBCATEGORY
    
    print(f"  Categories: {len(taxonomy)}")
    print(f"  Subcategories: {total_subcats}")
    print(f"  Expected examples: {expected_total:,}")
    print(f"  Validation samples: {EXTRA_EVAL_SAMPLES:,}")
    
    # Confirm before proceeding
    print(f"\nThis will make approximately {total_subcats} API calls to Claude Haiku 4.5.")
    print(f"Estimated cost: ~$0.50-1.50 (Haiku 4.5 pricing: $1/MTok input, $5/MTok output)")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Generate all captions
    all_data = generate_all_captions(api_key, taxonomy)
    
    if not all_data:
        print("\nError: No data was generated!")
        return
    
    # Split into train/valid
    print(f"\nSplitting into train/validation sets...")
    train_data, valid_data = split_train_valid(all_data, EXTRA_EVAL_SAMPLES)
    
    # Save to CSV
    print(f"\nSaving to CSV files...")
    save_to_csv(train_data, OUTPUT_TRAIN)
    save_to_csv(valid_data, OUTPUT_VALID)
    
    # Print statistics
    print_statistics(train_data, valid_data, taxonomy)
    
    print("\n✓ Data generation complete!")
    print(f"\nGenerated files:")
    print(f"  - {OUTPUT_TRAIN} (training data)")
    print(f"  - {OUTPUT_VALID} (validation data)")
    print(f"  - {OUTPUT_RAW_RESPONSES} (raw API responses log)")
    
    print(f"\nNext steps:")
    print(f"  1. Review the generated data in {OUTPUT_TRAIN}")
    print(f"  2. Update configs to point to these files:")
    print(f"     - configs/config_coarse.yaml: train_path: '{OUTPUT_TRAIN}'")
    print(f"     - configs/config_fine.yaml: train_path: '{OUTPUT_TRAIN}'")
    print(f"  3. Train models:")
    print(f"     python train.py --config configs/config_coarse.yaml")
    print(f"     python train.py --config configs/config_fine.yaml")


if __name__ == "__main__":
    main()

