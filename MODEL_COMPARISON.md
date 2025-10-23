# Model Comparison - VW vs Embedding

## Performance Summary

### Coarse Model (45 Categories)

| Model | Architecture | Training | Inference | Validation Accuracy |
|-------|-------------|----------|-----------|---------------------|
| VW (optimized) | Bag-of-words | 2 sec | <1ms | **41.55%** (831/2,000) |
| **Embedding** | Semantic | 3 sec* | 3-5ms | **48.35%** (967/2,000) ✓ |

*First run only. Cached runs: <1 sec

**Winner: Embedding (+6.80% improvement)**

### Fine Model (422 Subcategories)

| Model | Architecture | Training | Inference | Validation Accuracy |
|-------|-------------|----------|-----------|---------------------|
| VW (optimized) | Bag-of-words | 5 sec | <1ms | **15.50%** (310/2,000) |
| **Embedding** | Semantic | 3 sec* | 3-5ms | **30.85%** (617/2,000) ✓ |

*Cached run: <1 sec

**Winner: Embedding (+15.35% improvement, nearly DOUBLED!)**

## Architecture Comparison

### VW (Vowpal Wabbit)
```
Text → Tokenization → Hash Features → Linear Model → Prediction
      (split words)  (16M-128M dims)  (online SGD)    (<1ms)
```

**Pros:**
- Extremely fast inference (<1ms)
- Online learning capable
- Tiny memory footprint
- No GPU needed

**Cons:**
- Bag-of-words only (no semantics)
- Hash collisions
- Limited to n-grams

### Embedding (SentenceTransformers + Sklearn)
```
Text → Sentence Encoder → Dense Embeddings → Classifier → Prediction
      (BERT-based)       (384-dim vector)   (LogReg)    (3-5ms)
```

**Pros:**
- Semantic understanding
- Much better accuracy (+7% to +15%)
- Still very fast (<5ms)
- Cached embeddings (instant retraining)

**Cons:**
- Slightly slower inference (3-5ms vs <1ms)
- Larger model size (417KB vs VW's size)
- Requires sentence-transformers library

## Use Cases

### Use VW When:
- Need absolute minimum latency (<1ms critical)
- Online learning required (streaming updates)
- Extremely resource-constrained environment
- Keyword-based classification acceptable

### Use Embedding When:
- Accuracy is important
- Can afford 3-5ms inference time
- Semantic understanding needed
- Keywords avoided/not reliable
- **RECOMMENDED for this project**

## Model Files

### VW Models
```
models/model_coarse_best.vwbin (92 MB)    - 41.55% accuracy
models/model_fine_best.vwbin (708 MB)     - 15.50% accuracy
```

### Embedding Models
```
models/model_coarse_embedding.pkl (417 KB)  - 48.35% accuracy
models/model_fine_embedding.pkl (3.9 MB)    - 30.85% accuracy
```

### Embeddings Cache
```
data/embedding_cache/train_combined_all_MiniLM_L6_v2.npy (21.8 MB)
```

## Production Recommendation

**Use Embedding Models:**
1. 48% accuracy on 45 classes (vs 42% VW)
2. 31% accuracy on 422 classes (vs 16% VW)
3. Still very fast (<5ms inference)
4. Better for real-world video captions

The small speed trade-off (1ms → 5ms) is worth the significant accuracy gain.

## Hybrid Approach (Best of Both)

For maximum performance:

1. **First filter with embedding model** → Top-5 candidates
2. **Refine with LLM** → Final label

This gives you:
- Fast filtering (5ms per video)
- High accuracy (LLM on reduced candidates)
- Cost-effective (LLM only needed for ambiguous cases)

## Training Times

|  | VW | Embedding |
|--|----|-----------| 
| Coarse (first run) | 2 sec | 3 sec |
| Coarse (cached) | 2 sec | <1 sec |
| Fine (first run) | 5 sec | 3 sec |
| Fine (cached) | 5 sec | <1 sec |

**Embedding models are faster to retrain thanks to caching!**

## Conclusion

✓ **Embedding models win** for this use case:
- Significantly better accuracy
- Still very fast inference
- Semantic understanding  
- Better for production deployment

The Vorpal system now offers both options - use embedding models for best results!
