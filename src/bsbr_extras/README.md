# BSBR Extras

This package contains additional transformer architectures for evaluation and research purposes, as well as utilities for converting pre-trained models to use Block Sparse Attention with Block Retrieval (BSBR).

## GPT-to-BSBR Converter

The converter allows you to transform pre-trained GPT-2 models into BSBR models, preserving their knowledge while making them more efficient for long-context processing.

### Basic Usage

```python
from bsbr_extras import convert_to_bsbr

# Convert a standard GPT-2 model to BSBR
bsbr_model = convert_to_bsbr(
    model_name_or_path="gpt2",  # Can be any HuggingFace model name or local path
    chunk_size=128,             # Size of chunks for block processing
    compression_factor=4        # Optional: Compress chunk states
)

# Use the model
input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
attention_mask = torch.ones_like(input_ids)
outputs = bsbr_model(input_ids, attention_mask=attention_mask)
```

### Advanced Usage

For more control, you can use the `TransformerToBSBRConverter` class directly:

```python
from transformers import GPT2Model
from bsbr_extras import TransformerToBSBRConverter

# Load original model
original_model = GPT2Model.from_pretrained("gpt2")

# Initialize converter
converter = TransformerToBSBRConverter(
    chunk_size=128,
    compression_factor=4
)

# Convert model
bsbr_model = converter.convert_gpt2_model_to_bsbr(original_model)
```

### Performance Comparison Example

We provide a script to compare the performance of the original model and the converted BSBR model:

```bash
python -m bsbr_extras.convert_example --model_name gpt2 --chunk_size 128 --seq_len 1024
```

Arguments:
- `--model_name`: HuggingFace model name or path (default: "gpt2")
- `--chunk_size`: Size of chunks for BSBR processing (default: 128)
- `--compression_factor`: Optional compression factor for state vectors
- `--seq_len`: Sequence length for comparison (default: 1024)
- `--output_dir`: Directory to save the converted model (default: "./converted_model")
- `--save_model`: Flag to save the converted model

## Benefits of BSBR

BSBR provides several advantages over standard transformers:

1. **Better Scaling**: More efficient for longer sequences, with performance approaching O(n) in many cases
2. **Memory Efficiency**: Can process longer contexts with the same memory footprint
3. **Preserved Knowledge**: Retains the knowledge embedded in the pre-trained weights
4. **Information Retention**: Maintains important context through chunk-level retrieval

## Converting Additional Model Types

Currently, the converter supports GPT-2 models. Support for additional model architectures will be added in future releases.

## Mathematical Correspondence

The conversion from standard attention to BSBR attention exploits the following key insights:

1. Within chunks, the models process information identically
2. Between chunks, BSBR uses meta-queries and meta-keys to retrieve condensed state vectors
3. The Q, K, V projections are directly transferred from the original model
4. Meta-projection weights are initialized as combinations of the original projections

This approach preserves the model's learned patterns while introducing more efficient computation for long-range dependencies. 