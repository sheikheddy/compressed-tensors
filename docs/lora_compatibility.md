# LoRA Compatibility for INT4 Compressed Models

This document describes the `LoRACompatibleCompressedLinear` class, which enables LoRA (Low-Rank Adaptation) support for INT4/INT8 compressed models in vLLM and other inference engines.

## Problem Statement

vLLM's LoRA injection mechanism expects standard PyTorch tensors with shape `(out_features, in_features)`. However, compressed-tensors stores INT4 weights in a packed format:
- **Packed weights**: `weight_packed` (torch.int32) with shape `(out_features, in_features / 8)`
- **Quantization parameters**: `weight_scale`, `weight_zero_point`, `weight_shape`

When vLLM tries to inject LoRA adapters, it fails because:
1. No `weight` attribute exists (only `weight_packed`)
2. Packed buffers have incorrect shapes for matrix operations
3. INT4 values need dequantization before LoRA delta application

## Solution: LoRACompatibleCompressedLinear

The `LoRACompatibleCompressedLinear` class extends `CompressedLinear` to expose a `weight` property that:
1. **Lazy decompression**: Materializes float weights only when accessed
2. **LoRA compatibility**: Returns tensors with correct shape for LoRA injection
3. **Memory efficiency**: Caches materialized weights to avoid repeated decompression
4. **Flexible modes**: Supports both quantized inference and LoRA-merged inference

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LoRACompatibleCompressedLinear              │
├─────────────────────────────────────────────────────────────┤
│  Storage (packed):                                           │
│    • weight_packed: torch.int32 (out_feat, in_feat/8)      │
│    • weight_scale: scaling factors                          │
│    • weight_zero_point: zero points (optional)              │
│    • weight_shape: original shape metadata                  │
├─────────────────────────────────────────────────────────────┤
│  Property (materialized):                                    │
│    • weight: torch.Tensor (out_feat, in_feat)              │
│      └─> Lazy decompression: int32 → int8 → float          │
├─────────────────────────────────────────────────────────────┤
│  Modes:                                                      │
│    • Quantized: Decompress on-the-fly, no LoRA             │
│    • Materialized: Use cached weight + LoRA deltas          │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Conversion

```python
from compressed_tensors.linear import LoRACompatibleCompressedLinear
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
from compressed_tensors.config import CompressionFormat

# Convert a standard Linear layer to LoRA-compatible compressed format
lora_module = LoRACompatibleCompressedLinear.from_linear(
    module=linear_layer,
    quantization_scheme=QuantizationScheme(
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=128,
        )
    ),
    quantization_format=CompressionFormat.pack_quantized.value,
    materialize_on_init=False,  # Lazy materialization (default)
)
```

### For vLLM Integration

```python
# In vLLM's model loading code:
from compressed_tensors.linear import LoRACompatibleCompressedLinear

# When loading Kimi K2 or other INT4 compressed models:
# Replace CompressedLinear with LoRACompatibleCompressedLinear

# vLLM can now access weights for LoRA injection:
base_weight = module.weight  # Returns decompressed float tensor

# Apply LoRA delta (vLLM's LoRA injection code):
lora_A = ...  # (out_features, rank)
lora_B = ...  # (rank, in_features)
lora_scale = ...

merged_weight = base_weight + lora_scale * (lora_A @ lora_B)

# Update the module weight:
module.weight = merged_weight  # Cached for efficient forward passes
```

### Multi-Tenant LoRA Serving

```python
# Workshop Labs' use case: Hot-swapping LoRA adapters

def load_user_lora(module, user_id):
    """Load user-specific LoRA adapter."""
    # Get base weight (decompressed once, then cached)
    base_weight = module.weight

    # Load user's LoRA adapter
    lora_A, lora_B = load_lora_for_user(user_id)

    # Merge LoRA with base
    merged_weight = base_weight + (lora_A @ lora_B)

    # Update module
    module.weight = merged_weight
    return module

def unload_lora(module):
    """Unload LoRA and free memory."""
    module.clear_materialized_weight()
    # Next forward pass will use compressed weights (no LoRA)
```

### Kimi K2 Thinking Configuration

Kimi K2 uses INT4 with group-wise quantization (group_size=128):

```python
from compressed_tensors import QuantizationConfig, QuantizationScheme, QuantizationArgs

# Kimi K2-like configuration
kimi_config = QuantizationConfig(
    config_groups={
        "kimi_experts": QuantizationScheme(
            targets=["Linear"],  # MoE expert layers
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=128,
            ),
            format="pack-quantized",  # compressed-tensors INT4 format
        )
    }
)

# Convert model to LoRA-compatible format
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        lora_module = LoRACompatibleCompressedLinear.from_linear(
            module=module,
            quantization_scheme=kimi_config.config_groups["kimi_experts"],
            quantization_format="pack-quantized",
        )
        # Replace in model...
```

## API Reference

### LoRACompatibleCompressedLinear

#### Class Methods

**`from_linear(module, quantization_scheme, quantization_format, materialize_on_init=False)`**
- Converts a `torch.nn.Linear` module to LoRA-compatible compressed format
- Parameters:
  - `module`: Linear layer to compress
  - `quantization_scheme`: Quantization configuration
  - `quantization_format`: Compression format (e.g., "pack-quantized")
  - `materialize_on_init`: If True, decompress weights immediately; if False, use lazy materialization
- Returns: `LoRACompatibleCompressedLinear` instance

#### Properties

**`weight`** (getter)
- Returns materialized float weight tensor
- Lazy decompression on first access
- Cached for subsequent accesses
- Shape: `(out_features, in_features)`

**`weight`** (setter)
- Updates weight with merged LoRA deltas
- Marks module to use materialized weight in forward passes
- Example: `module.weight = base + lora_delta`

#### Instance Methods

**`forward(input: Tensor) -> Tensor`**
- Forward pass with automatic mode selection:
  - **Quantized mode**: Decompress on-the-fly from packed buffers
  - **Materialized mode**: Use cached weight (with LoRA deltas)

**`clear_materialized_weight()`**
- Clears cached materialized weight to free memory
- Useful when switching between LoRA adapters
- Next `weight` access will trigger re-decompression

**`get_quantization_params() -> Dict`**
- Returns quantization metadata for LoRA scaling:
  - `quantization_scheme`: Full quantization configuration
  - `weight_scale`: Scaling factors for dequantization
  - `weight_zero_point`: Zero points (if asymmetric)
  - `weight_g_idx`: Group indices (if group quantization)
  - `weight_shape`: Original weight shape

**`update_compressed_weight(new_weight: Tensor)`**
- (Future) Re-quantizes merged weight back to packed format
- Useful for persistent LoRA merges
- Currently stores in materialized form; re-quantization TODO

## Implementation Details

### Memory Overhead

**Packed storage** (INT4):
- `weight_packed`: 0.5 bytes per parameter (int32 with 8 values packed)
- `weight_scale`: 4 bytes per group (float32)
- Total: ~0.5-0.6 bytes per parameter

**Materialized storage** (BF16):
- `_materialized_weight`: 2 bytes per parameter
- **Overhead: ~4x vs packed**

**Example**: 14336 × 4096 MoE expert (Kimi K2)
- Packed: ~29 MB
- Materialized: ~117 MB
- Overhead: +88 MB per expert

For multi-tenant serving with hot-swapping, this is acceptable as only active adapters need materialization.

### Decompression Pipeline

```
weight_packed (int32)           [8 values per int32]
    ↓ unpack_from_int32()
weight (int8)                   [signed int8: -8 to 7]
    ↓ dequantize()
weight (float)                  [scaled and offset]
    ↓ apply LoRA
weight + lora_A @ lora_B        [merged weight]
    ↓ forward pass
output
```

### Performance Considerations

1. **First access latency**: Decompression takes ~10-50ms for large layers
2. **Subsequent accesses**: No overhead (cached)
3. **Memory**: 4x overhead for materialized weights
4. **LoRA adapter swapping**: Fast (just update cached weight)
5. **Forward pass**: Same performance as standard linear layer

## vLLM Integration Path

For Workshop Labs and vLLM contributors:

### Phase 1: Detection (vLLM side)
```python
# In vLLM's model loader:
if hasattr(module, 'weight_packed') and hasattr(module, 'weight_scale'):
    # Detected compressed-tensors INT4 format
    if needs_lora_support:
        # Use LoRACompatibleCompressedLinear wrapper
        module = LoRACompatibleCompressedLinear.from_linear(...)
```

### Phase 2: LoRA Injection (vLLM side)
```python
# In vLLM's LoRA adapter code:
def inject_lora_adapter(module, lora_A, lora_B, scale):
    if isinstance(module, LoRACompatibleCompressedLinear):
        base_weight = module.weight  # Triggers decompression
        merged = base_weight + scale * (lora_A @ lora_B)
        module.weight = merged
    else:
        # Standard LoRA injection
        ...
```

### Phase 3: FusedMoE Integration (vLLM side)
```python
# In FusedMoEPermuteExpertsUnpermute kernel:
def forward(self, x, expert_weights, ...):
    if has_lora_adapters:
        # Use materialized weights from LoRACompatibleCompressedLinear
        weights = [expert.weight for expert in experts]
    else:
        # Use compressed weights directly
        weights = [expert.weight_packed for expert in experts]

    # Run FusedMoE kernel...
```

## Testing

Tests are located in `tests/test_linear/test_lora_compatible_linear.py`:

```bash
pytest tests/test_linear/test_lora_compatible_linear.py -v
```

Test coverage includes:
- Basic conversion and materialization
- Lazy vs eager weight decompression
- Forward pass in quantized and materialized modes
- LoRA adapter simulation
- Group-wise quantization (Kimi K2 config)
- Memory management (clearing cached weights)

## Future Enhancements

1. **Re-quantization support**: Implement `update_compressed_weight()` to re-pack merged weights
2. **Kernel integration**: LoRA-aware quantized kernels (no materialization needed)
3. **Memory optimization**: Partial materialization (only changed experts)
4. **Profiling tools**: Memory and latency analysis for LoRA serving

## References

- [vLLM FusedMoE Architecture](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/)
- [compressed-tensors Documentation](https://github.com/vllm-project/compressed-tensors)
- [Workshop Labs Use Case](https://docs.google.com/document/d/19CsSgU_aPnYTwNoz67TN9Vdfba_EvlGX4TvRcOQ9Nzw/) (if accessible)

## Contact

For questions or contributions:
- Open an issue on [compressed-tensors GitHub](https://github.com/vllm-project/compressed-tensors/issues)
- vLLM community: [vLLM GitHub](https://github.com/vllm-project/vllm)
- Workshop Labs: luke@workshoplabs.ai
