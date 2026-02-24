# BDH Data Extraction Tools

A comprehensive suite of tools for extracting and analyzing data from the BDH (Bidirectional Deep Highway) model. This modular system extracts neuron activations, embeddings (with RoPE positional encoding), predictions, and raw logits from text input.

## Features

- **🧠 Neuron Extraction**: Capture neuron activations from all layers and heads
- **📊 Embedding Extraction**: Extract 256D embeddings, reduce to 30D/32D, includes RoPE transformation
- **🎯 Prediction Generation**: Predict top-K next tokens with probabilities
- **📈 Raw Logits**: Save complete logit distributions for post-processing
- **🔄 Prediction Updates**: Recalculate predictions with different temperature/top-K without re-running model
- **🗂️ Organized Output**: Clean directory structure with separate files for each data type
- **⚡ Modular Design**: Separate extraction modules for flexibility

## Quick Start

Run the main data extractor:
```bash
./run_demo.sh
```

Or with conda directly:
```bash
conda activate ml
python demo.py
```

Update predictions from existing logits (fast!):
```bash
./run_update_predictions.sh
```

## Tools Overview

### 1. `demo.py` - Complete Data Extractor
**Main tool** that orchestrates all extractions and saves to `Data/infer/` directory.

**What it generates:**
- `embeddings.csv` - Token embeddings (original + RoPE, 30D normalized)
- `predictions.csv` - Top-K next token predictions with probabilities
- `logits.csv` - Raw logits for all vocabulary tokens
- `layer1/` to `layer6/` - Neuron activations for each layer
  - `head1.csv` to `head4.csv` - Top 50 neurons per head

**Usage:**
```bash
./run_demo.sh
# Follow prompts:
# 1. Enter text: "Once upon a time"
# 2. Temperature (default 0.8)
# 3. Top-K (default 20)
```

### 2. `update_predictions.py` - Fast Prediction Updates
**Recalculates predictions** from existing logits without re-running the model (instant).

**Use case:** Experiment with different temperature/top-K values on same input.

**Usage:**
```bash
./run_update_predictions.sh
# Loads existing logits from Data/infer/logits.csv
# Enter new temperature and top-K values
# Updates Data/infer/predictions.csv
```

## File Structure

### Execution Scripts
- **`demo.py`** - Main orchestrator, imports all extraction modules
- **`update_predictions.py`** - Prediction recalculation tool
- **`run_demo.sh`** - Launcher script for demo.py (activates ml env)
- **`run_update_predictions.sh`** - Launcher script for update_predictions.py

### Extraction Modules
- **`extract_neurons.py`** - Neuron activation extraction for all layers/heads
- **`extract_embeddings.py`** - Embedding extraction with dimension reduction
- **`extract_predictions.py`** - Next token prediction generation
- **`extract_logits.py`** - Raw logit extraction and saving

### Utilities
- **`model_loader.py`** - BDH model and tokenizer initialization
- **`embedding_processing.py`** - Dimension reduction, normalization, RoPE transformation

## Output Structure

All outputs are saved to `Data/infer/`:

```
Data/infer/
├── embeddings.csv        # Token embeddings (original + RoPE)
├── predictions.csv       # Top-K predictions with probabilities
├── logits.csv           # Raw logits for all vocab tokens
├── layer1/
│   ├── head1.csv        # Top 50 neurons for layer 1, head 1
│   ├── head2.csv
│   ├── head3.csv
│   └── head4.csv
├── layer2/
│   └── ...
...
└── layer6/
    └── ...
```

## CSV Output Formats

### Embeddings CSV
Includes both original and RoPE-transformed embeddings:

```csv
Token_Text,Token_ID,Embedding_Type,Dim_0,Dim_1,...,Dim_29
Once,12966,original,0.4594,0.4635,...,0.2506
Once,12966,rope,0.5123,0.3821,...,0.3145
 upon,2402,original,0.7324,0.2248,...,0.4425
 upon,2402,rope,0.6891,0.3012,...,0.3987
```

**Note:** Each token has 2 rows - one for original embeddings, one for RoPE-transformed.

### Predictions CSV
```csv
Input_Text,Once upon a time
Temperature,0.8
Top_K,20

Rank,Token,Token_ID,Probability
1, little,1310,0.733400
2, boy,2933,0.124500
3, girl,2576,0.089200
```

### Logits CSV
```csv
Input_Text,Once upon a time
Vocab_Size,50257

Token_ID,Logit
0,-15.234560
1,-12.456789
2,-18.901234
...
```

### Neuron CSV
```csv
Layer,Head,Neuron_Number,Value
1,1,4523,124.56
1,1,2891,98.34
1,1,7654,87.23
```

## Parameters

### Temperature (0.1 - 2.0)
- **0.1-0.5**: Focused, deterministic predictions
- **0.6-1.0**: Balanced (recommended: 0.8)
- **1.1-2.0**: Random, creative predictions

### Top-K (1 - 100)
- **1-10**: Most confident predictions only
- **10-30**: Good balance for analysis (recommended: 20)
- **30-100**: Comprehensive view

## Technical Details

### RoPE (Rotary Positional Embeddings)
Positional information is injected through rotation transformations rather than additive embeddings:

**Formula:** `v' = v * cos(pos × freq) + v_rot * sin(pos × freq)`

where position-dependent phases rotate embedding dimensions in pairs.

### Model Architecture
- **Layers**: 6
- **Heads per layer**: 4
- **Embedding dimension**: 256D
- **Reduced dimensions**: 30D (demo.py) or 32D (extract_embeddings.py)
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)

### Processing Pipeline
1. **Load Model** - Initialize BDH model and tokenizer
2. **Tokenize** - Text → Token IDs (tiktoken GPT-2)
3. **Extract Embeddings** - Token IDs → 256D embeddings
4. **Apply RoPE** - Rotary positional encoding transformation
5. **Reduce Dimensions** - 256D → 30D/32D (evenly spaced sampling)
6. **Normalize** - Min-max scaling to [0,1] range
7. **Generate Predictions** - Forward pass → Top-K tokens
8. **Extract Neurons** - Capture activations from each layer/head
9. **Save All** - Organized CSV outputs
### Processing Pipeline
1. **Load Model** - Initialize BDH model and tokenizer
2. **Tokenize** - Text → Token IDs (tiktoken GPT-2)
3. **Extract Embeddings** - Token IDs → 256D embeddings
4. **Apply RoPE** - Rotary positional encoding transformation
5. **Reduce Dimensions** - 256D → 30D/32D (evenly spaced sampling)
6. **Normalize** - Min-max scaling to [0,1] range
7. **Generate Predictions** - Forward pass → Top-K tokens
8. **Extract Neurons** - Capture activations from each layer/head
9. **Save All** - Organized CSV outputs

## Advanced Usage

### Using Extraction Modules Directly

```python
from model_loader import load_model, setup_tokenizer
from extract_embeddings import extract_embeddings_simple
from extract_predictions import extract_predictions
from extract_neurons import extract_all_neurons

# Load model once
model, config = load_model()
encoder = setup_tokenizer()

# Extract specific data
text = "Once upon a time"

# Just embeddings
token_ids, embeddings = extract_embeddings_simple(
    model, encoder, text, "output.csv"
)

# Just predictions
predictions, logits = extract_predictions(
    model, encoder, text, "pred.csv", temperature=0.9, top_k=10
)

# Just neurons
extract_all_neurons(model, config, encoder, text, "output_dir/")
```

### Customizing Extraction

Each module can be imported and customized:

```python
from extract_neurons import forward_with_neuron_capture, save_neurons_for_head

# Capture specific layer
result = forward_with_neuron_capture(model, config, token_tensor, capture_layer=3)
activations = result['activations']

# Save specific head
save_neurons_for_head(
    "layer3_head2.csv",
    text="Hello world",
    token_strs=["Hello", " world"],
    layer_num=3,
    head_num=1,  # 0-indexed
    activations=activations,
    config=result['config'],
    top_n=100  # Save top 100 neurons instead of 50
)
```

## Workflow Examples

### Example 1: Parameter Sweep
```bash
# Run model once
./run_demo.sh
# Input: "The cat sat on the"
# Temperature: 0.8, Top-K: 20

# Try different parameters instantly
./run_update_predictions.sh
# Temperature: 0.1  → deterministic
# Temperature: 1.5  → creative
# Temperature: 2.0  → very random
```

### Example 2: Batch Processing
```python
from model_loader import load_model, setup_tokenizer
from extract_predictions import extract_predictions

model, config = load_model()
encoder = setup_tokenizer()

texts = [
    "Once upon a time",
    "The quick brown fox",
    "In a galaxy far far away"
]

for i, text in enumerate(texts):
    extract_predictions(
        model, encoder, text,
        f"pred_{i}.csv",
        temperature=0.8,
        top_k=20
    )
```

## Dependencies

### Required
- **Python**: 3.8+
- **PyTorch**: For model operations
- **tiktoken**: GPT-2 tokenizer
- **NumPy**: Array operations

### Model
- **BDH Checkpoint**: `../Models/BDH_model/checkpoints/ckpt_5000.pt`

## Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named 'torch'`**
- Activate ML environment: `conda activate ml`
- Or use launcher scripts: `./run_demo.sh`

**2. Model not found**
- Check path: `../Models/BDH_model/checkpoints/ckpt_5000.pt`
- Ensure checkpoint exists

**3. Permission denied on .sh files**
- Make executable: `chmod +x run_demo.sh run_update_predictions.sh`

**4. Output directory not created**
- The script auto-creates `Data/infer/` - check write permissions

### Performance Tips

- **Model loading**: Model loads once per session (2GB RAM)
- **Fast updates**: Use `update_predictions.py` to avoid reloading model
- **Batch processing**: Load model once, process multiple texts
- **Memory**: Close other applications if needed

## Output File Sizes

Approximate sizes for typical inputs:

- **embeddings.csv**: ~5KB per token (includes original + RoPE)
- **predictions.csv**: ~2KB (top-20)
- **logits.csv**: ~1.2MB (full vocabulary, 50K+ tokens)
- **neuron CSV**: ~10KB per head (top-50 neurons)
- **Total per run**: ~1.5MB for 4-token input

## License

Part of the BDH_visuals project for educational and research purposes.

## Contact & Issues

For issues or questions, refer to the main project repository.