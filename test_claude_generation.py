#!/usr/bin/env python3
"""
Test the Claude API caption generation with a few examples before running the full dataset.

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python test_claude_generation.py
"""

import json
import os
import sys

try:
    import anthropic
except ImportError:
    print("Error: Please install anthropic library:")
    print("  pip install anthropic")
    sys.exit(1)


def test_generation():
    """Test caption generation for a few subcategories."""
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Test cases
    test_cases = [
        ("Sports & Fitness", "Soccer Matches in Stadiums"),
        ("Nature & Wildlife", "Tropical Rainforests & Jungle Wildlife"),
        ("Technology & Computing", "Programmers & Coding Screens"),
        ("Food & Culinary", "Street Food Stalls & Night Markets"),
    ]
    
    print("="*70)
    print("TESTING CLAUDE CAPTION GENERATION")
    print("="*70)
    
    for category, subcategory in test_cases:
        print(f"\n{'─'*70}")
        print(f"Category: {category}")
        print(f"Subcategory: {subcategory}")
        print(f"{'─'*70}")
        
        # Extract keywords to avoid
        combined = f"{category} {subcategory}".lower()
        words = combined.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace('-', ' ').replace('&', ' ').split()
        stop_words = {'and', 'the', 'a', 'an', 'of', 'in', 'at', 'to', 'for', 'with', 'on'}
        keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
        keywords_str = ", ".join(keywords)
        
        prompt = f"""You are generating realistic video descriptions for a video classification dataset.

Category: {category}
Subcategory: {subcategory}

Generate 5 diverse, realistic video descriptions that would fit this subcategory.

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

Generate 5 unique descriptions now:"""
        
        print(f"\nKeywords to avoid: {keywords_str}")
        print(f"\nGenerating captions...")
        
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",  # Claude Haiku 4.5 - fastest and most cost-efficient
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text
            
            # Parse response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    captions = json.loads(json_str)
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    captions = json.loads(json_str)
                else:
                    captions = json.loads(response)
                
                print(f"\nGenerated {len(captions)} captions:")
                for i, caption in enumerate(captions, 1):
                    print(f"  {i}. {caption}")
                    
                    # Check if keywords were avoided
                    caption_lower = caption.lower()
                    found_keywords = [kw for kw in keywords if kw in caption_lower]
                    if found_keywords:
                        print(f"     ⚠️  Contains keywords: {', '.join(found_keywords)}")
            
            except json.JSONDecodeError as e:
                print(f"\n⚠️  Failed to parse JSON response:")
                print(response[:500])
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print("Test complete! If the results look good, run:")
    print("  python generate_data_with_claude.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_generation()

