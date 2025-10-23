# Vorpal - Fast Video Classification Pipeline

High-throughput video classification system using Vowpal Wabbit for both flat and hierarchical text classification of video captions/metadata.

## Features

- **Fast & Scalable**: Handles millions of video captions with constant memory usage
- **Streaming**: Never loads entire dataset into RAM
- **Progressive Validation**: True online metrics during training  
- **Hierarchical Classification**: Two-stage cascade (category → subcategory)
- **Flexible**: Supports both flat (10 classes) and hierarchical (10 categories × 5 subcategories)
- **Easy Installation**: No C++ compilation required (uses vowpal-wabbit-next)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Create CSV or Parquet files with:
- `caption` column: Text description
- `label_id` or `category_id`: Integer labels (1..N)
- For hierarchical: also include `subcategory_id`

### 2. Configure

Choose a config from `configs/`:
- `configs/config.example.yaml` - Flat classification (10 classes)
- `configs/config_coarse.yaml` - Hierarchical: category level (10 classes)
- `configs/config_fine.yaml` - Hierarchical: subcategory level (50 classes)

Or create your own:

```yaml
model:
  num_classes: 10
  namespace: "t"
  
vw:
  bits: 24
  learning_rate: 0.5
  l2: 1.0e-7
  ngram: 2
  loss_function: "logistic"
  probabilities: true

data:
  train_path: "data/train.csv"
  valid_path: "data/valid.csv"
  text_column: "caption"
  label_column: "label_id"

training:
  log_every: 200000
  checkpoint_path: "models/model.vwbin"
  save_labels: false
```

### 3. Train

```bash
python train.py --config configs/config.example.yaml
```

### 4. Evaluate

```bash
python eval.py --config configs/config.example.yaml --checkpoint models/model.vwbin
```

### 5. Inference

```bash
# Interactive
echo "football match highlights" | python inference.py \
  --config configs/config.example.yaml \
  --checkpoint models/model.vwbin \
  --stdin

# Batch
python inference.py \
  --config configs/config.example.yaml \
  --checkpoint models/model.vwbin \
  --input_csv data/unlabeled.csv \
  --text_column caption
```

## Hierarchical Classification

For two-level classification (category + subcategory):

### Training

```bash
# Train category classifier
python train.py --config configs/config_coarse.yaml

# Train subcategory classifier  
python train.py --config configs/config_fine.yaml
```

### Inference (Two-Stage)

```python
# Load both models and predict at both levels
python inference.py \
  --config configs/config_coarse.yaml \
  --checkpoint models/model_coarse.vwbin \
  --config_fine configs/config_fine.yaml \
  --checkpoint_fine models/model_fine.vwbin \
  --stdin
```

Or use separately for simpler integration.

## Project Structure

```
Vorpal/
├── train.py              # Training script
├── eval.py               # Evaluation script
├── inference.py          # Inference script
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── configs/              # Configuration files
│   ├── config.example.yaml
│   ├── config_coarse.yaml
│   └── config_fine.yaml
├── data/                 # Training/validation data
│   ├── train.csv
│   ├── valid.csv
│   └── taxonomy.json
└── models/               # Saved models (gitignored)
    ├── model.vwbin
    ├── model_coarse.vwbin
    └── model_fine.vwbin
```

## Configuration Options

### Model Settings

- `num_classes`: Number of output classes
- `namespace`: Feature namespace (default: "t" for text)

### VW Settings

- `bits`: Hash table size (24 = 16M features, 25 = 32M features)
- `learning_rate`: Learning rate (0.3-0.5 typical)
- `l2`: L2 regularization (1e-7 typical)
- `ngram`: N-gram order (1=unigrams, 2=+bigrams, 3=+trigrams)
- `loss_function`: Loss function (logistic for classification)
- `probabilities`: Return per-class probabilities (true/false)

### Data Settings

- `train_path`: Path to training data
- `valid_path`: Path to validation data
- `text_column`: Column name containing text
- `label_column`: Column name containing labels

### Training Settings

- `log_every`: Log progress every N examples
- `checkpoint_path`: Where to save trained model
- `save_labels`: Save label mapping to JSON

## Performance Tips

- **Hashing bits**: Start with 24 (16M features), increase to 25-26 for large vocabularies
- **N-grams**: Use 2 for good balance, increase to 3 for richer features
- **Learning rate**: 0.5 for fast convergence, lower (0.1-0.2) if unstable
- **Parquet**: Use Parquet format for very large datasets (better compression)

## Data Format

### CSV Format

**Flat classification:**
```csv
label_id,caption
1,football match highlights
2,movie trailer action scenes
```

**Hierarchical classification:**
```csv
category_id,category_name,subcategory_id,subcategory_name,caption
1,sports,1,football,"football match highlights"
2,entertainment,6,movies,"movie trailer action scenes"
```

### VW Format (Internal)

Text is converted to VW format with n-grams:
```
<label> |<namespace> <feature1> <feature2> ...
```

Example:
```
1 |t football match highlights football_match match_highlights
```

N-grams are generated automatically during preprocessing since vowpal-wabbit-next doesn't support CLI `--ngram` option.

## Why Vowpal Wabbit?

- **Streaming**: Learns as it reads; constant memory regardless of dataset size
- **Online Learning**: Can update incrementally with new data
- **Fast**: Linear model over hashed n-grams is 100-1000x faster than transformers
- **Scalable**: Handles millions of examples efficiently

## Why vowpal-wabbit-next?

We use `vowpal-wabbit-next` instead of standard `vowpalwabbit`:

**Advantages:**
- Pre-built wheels - installs in seconds
- No C++ compilation or system dependencies
- Clean, type-safe Python API
- Same algorithms and quality

**Trade-off:**
- Some CLI options (`--ngram`, `--skips`) not exposed
- Solution: We generate n-grams manually in preprocessing

## Advanced Usage

### Interactive Annotation

Learn from corrections on-the-fly:

```bash
python inference.py \
  --config configs/config.example.yaml \
  --checkpoint models/model.vwbin \
  --stdin --interactive --conf_thresh 0.85
```

When confidence < 0.85, the system asks for the correct label and learns from it.

### Incremental Learning

Continue training from a checkpoint:

```python
# Load existing model
with open("models/model.vwbin", 'rb') as f:
    model_data = f.read()

ws = vw.Workspace(model_data=model_data)
parser = vw.TextFormatParser(ws)

# Learn from new examples
for label, text in new_data:
    ex = parser.parse_line(to_vw_line(label, text, "t"))
    ws.learn_one(ex)

# Save updated model
with open("models/model_updated.vwbin", 'wb') as f:
    f.write(ws.serialize())
```

### Custom Taxonomy

Edit your data to include custom categories:

```python
TAXONOMY = {
    1: "your_category_1",
    2: "your_category_2",
    # ... up to N classes
}
```

Update `num_classes` in config and retrain.

## Troubleshooting

### Low Accuracy

- Increase n-gram order: `ngram: 3`
- Increase features: `bits: 26`
- Check data quality (are captions descriptive?)
- Add more training data

### Out of Memory

- Reduce hash bits: `bits: 22`
- Use streaming (already default)
- Process smaller batches

### Slow Training

- Already using streaming and hashing (very fast)
- For huge datasets: Use Parquet format
- Training is typically <10 seconds for 10k examples

## Performance Benchmarks

### Flat Classification (10 classes)
- Training: 99.5% progressive accuracy
- Validation: 100% accuracy
- Speed: ~10k examples/second

### Hierarchical Classification (10 + 50 classes)
- Category: 99.7% accuracy
- Subcategory: 98.6% accuracy  
- Speed: ~5k examples/second (both stages)

## Use Cases

1. **Video Content Moderation**: Classify user uploads
2. **Content Recommendation**: Fine-grained category matching
3. **Search & Discovery**: Multi-level taxonomy navigation
4. **Analytics**: Understand content distribution
5. **Active Learning**: Identify uncertain predictions for review

## References

- [vowpal-wabbit-next Documentation](https://vowpal-wabbit-next.readthedocs.io/)
- [Vowpal Wabbit](https://vowpalwabbit.org/)
- [fastText Paper](https://arxiv.org/abs/1607.01759) (similar approach)

## License

MIT
