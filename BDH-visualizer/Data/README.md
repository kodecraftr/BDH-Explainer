# BDH Data Extraction Tools

This directory contains a comprehensive suite of tools designed for extracting and analyzing internal states from the Baby Dragon Hatchling (BDH) model. This modular system facilitates the extraction of neuron activations, topological embeddings (including Rotary Positional Encoding transformations), sequence predictions, and raw logit distributions.

---

## Capabilities

*   **Neuron Extraction (`extract_neurons.py`)**: Capture granular neuron activations across all model layers and attention heads.
*   **Embedding Extraction (`extract_embeddings.py`)**: Extract high-dimensional (256D) token embeddings and perform dimensionality reduction (30D/32D) inclusive of RoPE (Rotary Positional Encoding) phase transformations.
*   **Prediction Generation (`extract_predictions.py`)**: Generate top-K next-token predictions complete with probability distributions.
*   **Raw Logit Harvesting (`extract_logits.py`)**: Serialize complete vocabulary logit distributions for offline post-processing.
*   **Fast Prediction Re-sampling (`update_predictions.py`)**: Recalculate token predictions instantly under varied temperature and top-K constraints without requiring a full model forward pass.

---

## Quick Start Guide

**1. Complete Data Extraction Pipeline:**

The primary orchestration script runs all extraction modules sequentially and outputs to the `Data/infer/` directory.

```bash
# Execute the automated environment wrapper
./run_demo.sh

# Alternatively, execute directly within the active Conda environment:
# conda activate ml
# python demo.py
```

*During execution, you will be prompted to provide an input text sequence, followed by desired Temperature (default: 0.8) and Top-K (default: 20) parameters.*

**2. Fast Prediction Updates:**

To experiment with different sampling parameters on an existing set of harvested logits:

```bash
./run_update_predictions.sh
```

---

## Output Architecture (`Data/infer/`)

Execution of the `demo.py` orchestrator populates the `Data/infer/` directory structured as follows:

```text
Data/infer/
├── embeddings.csv        # Token embeddings (Original + RoPE geometries)
├── predictions.csv       # Top-K sequence predictions and probabilities
├── logits.csv            # Raw logits covering the entire vocabulary space
├── layer1/
│   ├── head1.csv         # Top 50 activating neurons (Layer 1, Head 1)
│   ├── head2.csv
│   └── ...
...
└── layer6/
    └── ...
```

### Data Formats

**Embeddings CSV (`embeddings.csv`)**
Captures both spatial (original) and phase-rotated (RoPE) topological states mapping down to 30 continuous dimensions.
```csv
Token_Text,Token_ID,Embedding_Type,Dim_0,Dim_1,...,Dim_29
Once,12966,original,0.4594,0.4635,...,0.2506
Once,12966,rope,0.5123,0.3821,...,0.3145
```

**Predictions CSV (`predictions.csv`)**
```csv
Input_Text,Once upon a time
Temperature,0.8
Top_K,20

Rank,Token,Token_ID,Probability
1, little,1310,0.733400
2, boy,2933,0.124500
```

**Neuron CSV (`headX.csv`)**
```csv
Layer,Head,Neuron_Number,Value
1,1,4523,124.56
1,1,2891,98.34
```

---

## Sampling Parameters

### Temperature [0.1 - 2.0]
*   **0.1 - 0.5**: Highly deterministic, focused distribution. Recommended for standard language modeling tasks.
*   **0.6 - 1.0**: Balanced distribution. (Default setting: 0.8).
*   **1.1 - 2.0**: High variance, producing highly creative or unstable sequences.

### Top-K [1 - 100]
*   **1 - 10**: Restricts sampling strictly to highest confidence outputs.
*   **10 - 30**: Optimal range for generalized analysis. (Default setting: 20).
*   **30 - 100**: Comprehensive distributional view for deep behavior analysis.

---

## Extractor API & Advanced Usage

For custom integration workflows, individual extraction modules can be instantiated directly via Python scripts.

```python
from model_loader import load_model, setup_tokenizer
from extract_predictions import extract_predictions

# 1. Initialize the BDH Model and GPT-2 Tokenizer schema
model, config = load_model()
encoder = setup_tokenizer()

# 2. Extract Top-10 predictions using a creative temperature threshold
predictions, logits = extract_predictions(
    model=model, 
    encoder=encoder, 
    text="Once upon a time", 
    output_filename="pred.csv", 
    temperature=0.9, 
    top_k=10
)
```

For advanced node-level (neuron) captures, the core forward pass hook can intercept activation tensors dynamically:

```python
from extract_neurons import forward_with_neuron_capture

# Capture activation tensors strictly for Layer 3
result = forward_with_neuron_capture(model, config, token_tensor, capture_layer=3)
layer_3_activations = result['activations']
```

---

## Troubleshooting & Dependencies

**Primary Dependencies**
*   **Python:** 3.8+
*   **PyTorch:** Matrix operations and model graph parsing.
*   **tiktoken:** GPT-2 BPE (Byte-Pair Encoding) implementation.
*   **Expected Checkpoint Route:** Continually verifies against `../Models/BDH_model/checkpoints/ckpt_5000.pt`. Adjust pathing in scripts or via environment variables if checkpoints deviate.

**Common Resolution Paths**
*   *ModuleNotFoundError ('torch')*: Ensure the `ml` Conda environment is active before invoking Python explicitly.
*   *Permission Denied*: Ensure execution rights on shell wrappers: `chmod +x *.sh`.
*   *Memory Overflow*: The checkpoint loader inherently buffers roughly ~2GB into system RAM upon initialization. Ensure adequate hardware overhead before invoking batch tasks.