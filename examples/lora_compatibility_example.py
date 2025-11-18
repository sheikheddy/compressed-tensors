#!/usr/bin/env python3
"""
Example: Using LoRACompatibleCompressedLinear for INT4 + LoRA inference

This example demonstrates how to:
1. Create an INT4 compressed linear layer
2. Convert it to LoRA-compatible format
3. Simulate LoRA adapter injection
4. Perform inference with merged weights

Use case: Multi-tenant LoRA serving (like Workshop Labs' Kimi K2 deployment)
"""

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.linear import LoRACompatibleCompressedLinear
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
)


def create_int4_config(group_size=128, symmetric=True):
    """
    Create INT4 quantization config (similar to Kimi K2).

    Args:
        group_size: Size of quantization groups (128 for Kimi K2)
        symmetric: Use symmetric quantization (no zero-point)

    Returns:
        QuantizationConfig for INT4 W4A16 quantization
    """
    return QuantizationConfig(
        config_groups={
            "int4_group": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=symmetric,
                    strategy="group",
                    group_size=group_size,
                ),
            )
        }
    )


def simulate_lora_adapter(in_features, out_features, rank=8, scale=1.0):
    """
    Create a simulated LoRA adapter.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (typically 8-64)
        scale: LoRA scaling factor

    Returns:
        Tuple of (lora_A, lora_B, scale)
    """
    lora_A = torch.randn(out_features, rank) * 0.01
    lora_B = torch.randn(rank, in_features) * 0.01
    return lora_A, lora_B, scale


def main():
    print("=" * 70)
    print("LoRA-Compatible INT4 Compression Example")
    print("=" * 70)

    # Configuration
    in_features = 4096
    out_features = 14336  # Typical MoE expert size
    batch_size = 4
    lora_rank = 16

    print(f"\nConfiguration:")
    print(f"  Input features:  {in_features}")
    print(f"  Output features: {out_features}")
    print(f"  Batch size:      {batch_size}")
    print(f"  LoRA rank:       {lora_rank}")

    # Step 1: Create a standard Linear layer
    print("\n" + "-" * 70)
    print("Step 1: Create standard Linear layer")
    print("-" * 70)

    torch.manual_seed(42)
    linear_layer = torch.nn.Linear(in_features, out_features, bias=True)
    print(f"✓ Created Linear layer: {linear_layer}")

    # Calculate original size
    original_size_mb = (
        linear_layer.weight.numel() * linear_layer.weight.element_size()
        + linear_layer.bias.numel() * linear_layer.bias.element_size()
    ) / (1024**2)
    print(f"  Original size: {original_size_mb:.2f} MB")

    # Step 2: Apply INT4 quantization
    print("\n" + "-" * 70)
    print("Step 2: Apply INT4 quantization (group_size=128)")
    print("-" * 70)

    quant_config = create_int4_config(group_size=128, symmetric=True)
    model = torch.nn.Sequential(linear_layer)
    apply_quantization_config(model, quant_config)
    print("✓ Quantization config applied")

    # Step 3: Convert to LoRA-compatible format
    print("\n" + "-" * 70)
    print("Step 3: Convert to LoRA-compatible compressed format")
    print("-" * 70)

    lora_module = LoRACompatibleCompressedLinear.from_linear(
        module=linear_layer,
        quantization_scheme=quant_config.config_groups["int4_group"],
        quantization_format=CompressionFormat.pack_quantized.value,
        materialize_on_init=False,  # Lazy materialization
    )
    print(f"✓ Converted to LoRACompatibleCompressedLinear")

    # Check compressed size
    packed_size_mb = (
        lora_module.weight_packed.numel() * lora_module.weight_packed.element_size()
    ) / (1024**2)
    print(f"  Compressed size: {packed_size_mb:.2f} MB")
    print(f"  Compression ratio: {original_size_mb / packed_size_mb:.2f}x")

    # Check that weight is not materialized yet
    print(f"  Weight materialized: {lora_module._materialized_weight is not None}")

    # Step 4: Run baseline inference (no LoRA)
    print("\n" + "-" * 70)
    print("Step 4: Baseline inference (no LoRA)")
    print("-" * 70)

    input_tensor = torch.randn(batch_size, in_features)
    with torch.no_grad():
        output_baseline = lora_module(input_tensor)

    print(f"✓ Forward pass completed")
    print(f"  Input shape:  {input_tensor.shape}")
    print(f"  Output shape: {output_baseline.shape}")
    print(f"  Output mean:  {output_baseline.mean().item():.4f}")
    print(f"  Output std:   {output_baseline.std().item():.4f}")

    # Step 5: Access weight (triggers materialization)
    print("\n" + "-" * 70)
    print("Step 5: Access weight property (lazy decompression)")
    print("-" * 70)

    base_weight = lora_module.weight
    print(f"✓ Weight decompressed and cached")
    print(f"  Weight shape: {base_weight.shape}")
    print(f"  Weight dtype: {base_weight.dtype}")
    print(f"  Weight materialized: {lora_module._materialized_weight is not None}")

    materialized_size_mb = (
        base_weight.numel() * base_weight.element_size()
    ) / (1024**2)
    print(f"  Materialized size: {materialized_size_mb:.2f} MB")
    print(
        f"  Memory overhead: {materialized_size_mb / packed_size_mb:.2f}x vs compressed"
    )

    # Step 6: Create and apply LoRA adapter
    print("\n" + "-" * 70)
    print("Step 6: Create and apply LoRA adapter")
    print("-" * 70)

    lora_A, lora_B, lora_scale = simulate_lora_adapter(
        in_features, out_features, rank=lora_rank, scale=1.0
    )
    print(f"✓ Created LoRA adapter")
    print(f"  LoRA A shape: {lora_A.shape}")
    print(f"  LoRA B shape: {lora_B.shape}")
    print(f"  LoRA scale:   {lora_scale}")

    # Merge LoRA with base weights
    lora_delta = lora_scale * (lora_A @ lora_B)
    merged_weight = base_weight + lora_delta

    print(f"  LoRA delta mean: {lora_delta.mean().item():.6f}")
    print(f"  LoRA delta std:  {lora_delta.std().item():.6f}")

    # Update module with merged weight
    lora_module.weight = merged_weight
    print(f"✓ LoRA adapter applied to module")

    # Step 7: Run inference with LoRA
    print("\n" + "-" * 70)
    print("Step 7: Inference with LoRA adapter")
    print("-" * 70)

    with torch.no_grad():
        output_lora = lora_module(input_tensor)

    print(f"✓ Forward pass with LoRA completed")
    print(f"  Output shape: {output_lora.shape}")
    print(f"  Output mean:  {output_lora.mean().item():.4f}")
    print(f"  Output std:   {output_lora.std().item():.4f}")

    # Compare outputs
    output_diff = (output_lora - output_baseline).abs()
    print(f"\nComparison (LoRA vs baseline):")
    print(f"  Mean absolute diff: {output_diff.mean().item():.4f}")
    print(f"  Max absolute diff:  {output_diff.max().item():.4f}")

    # Step 8: Simulate adapter hot-swapping
    print("\n" + "-" * 70)
    print("Step 8: Simulate LoRA adapter hot-swapping")
    print("-" * 70)

    # Clear current adapter
    lora_module.clear_materialized_weight()
    print("✓ Cleared LoRA adapter (freed memory)")
    print(f"  Weight materialized: {lora_module._materialized_weight is not None}")

    # Apply different adapter
    lora_A2, lora_B2, _ = simulate_lora_adapter(
        in_features, out_features, rank=lora_rank, scale=0.5
    )
    base_weight = lora_module.weight  # Re-materialize base
    merged_weight2 = base_weight + 0.5 * (lora_A2 @ lora_B2)
    lora_module.weight = merged_weight2
    print("✓ Applied new LoRA adapter")

    # Run inference with new adapter
    with torch.no_grad():
        output_lora2 = lora_module(input_tensor)

    print(f"  Output mean:  {output_lora2.mean().item():.4f}")
    print(f"  Output std:   {output_lora2.std().item():.4f}")

    # Step 9: Get quantization parameters
    print("\n" + "-" * 70)
    print("Step 9: Access quantization metadata")
    print("-" * 70)

    quant_params = lora_module.get_quantization_params()
    print(f"✓ Retrieved quantization parameters:")
    print(f"  Available params: {list(quant_params.keys())}")
    print(
        f"  Weight scale shape: {quant_params['weight_scale'].shape if 'weight_scale' in quant_params else 'N/A'}"
    )
    print(
        f"  Weight shape: {quant_params['weight_shape'].tolist() if 'weight_shape' in quant_params else 'N/A'}"
    )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Successfully demonstrated LoRA-compatible INT4 compression")
    print(f"✓ Compression ratio: {original_size_mb / packed_size_mb:.2f}x")
    print(
        f"✓ Memory overhead (materialized): {materialized_size_mb / packed_size_mb:.2f}x"
    )
    print(f"✓ LoRA adapter hot-swapping: Working")
    print(f"\nReady for vLLM integration!")


if __name__ == "__main__":
    main()
