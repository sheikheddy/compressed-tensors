# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.linear.lora_compatible_linear import (
    LoRACompatibleCompressedLinear,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from torch.nn import Linear


def get_int4_quant_config(symmetric=True) -> QuantizationConfig:
    """Create INT4 W4A16 quantization config for testing."""
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=symmetric,
                strategy="tensor",
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


def get_int4_group_quant_config(group_size=128, symmetric=True) -> QuantizationConfig:
    """Create INT4 group-wise quantization config (like Kimi K2)."""
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=symmetric,
                strategy="group",
                group_size=group_size,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


class TestLoRACompatibleLinearBasic:
    """Test basic functionality of LoRACompatibleCompressedLinear."""

    @pytest.fixture
    def dummy_linear(self):
        """Create a dummy linear layer for testing."""
        torch.manual_seed(42)
        return Linear(128, 256, bias=True)

    @pytest.fixture
    def quant_config(self):
        """Default quantization config."""
        return get_int4_quant_config(symmetric=True)

    def test_from_linear_conversion(self, dummy_linear, quant_config):
        """Test that from_linear properly converts Linear to LoRACompatibleCompressedLinear."""
        # Apply quantization config
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        # Convert to LoRA-compatible
        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Verify it's the right type
        assert isinstance(lora_module, LoRACompatibleCompressedLinear)

        # Verify compressed parameters exist
        assert hasattr(lora_module, "weight_packed")
        assert hasattr(lora_module, "weight_scale")
        assert hasattr(lora_module, "weight_shape")

        # Verify status
        assert lora_module.quantization_status == QuantizationStatus.COMPRESSED

    def test_weight_materialization_lazy(self, dummy_linear, quant_config):
        """Test that weight materialization is lazy (only on first access)."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
            materialize_on_init=False,  # Lazy materialization
        )

        # Weight should not be materialized yet
        assert lora_module._materialized_weight is None

        # Access weight property
        materialized_weight = lora_module.weight

        # Now it should be materialized
        assert lora_module._materialized_weight is not None
        assert isinstance(materialized_weight, torch.Tensor)

        # Check shape is correct (decompressed shape)
        expected_shape = lora_module.weight_shape.tolist()
        assert list(materialized_weight.shape) == expected_shape

    def test_weight_materialization_eager(self, dummy_linear, quant_config):
        """Test that weight can be materialized on initialization."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
            materialize_on_init=True,  # Eager materialization
        )

        # Weight should be materialized immediately
        assert lora_module._materialized_weight is not None

    def test_weight_property_caching(self, dummy_linear, quant_config):
        """Test that weight property caches decompressed weights."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # First access
        weight1 = lora_module.weight
        # Second access
        weight2 = lora_module.weight

        # Should return the same cached tensor (identity check)
        assert weight1 is weight2


class TestLoRACompatibleLinearForward:
    """Test forward pass functionality."""

    @pytest.fixture
    def dummy_linear(self):
        torch.manual_seed(42)
        return Linear(128, 256, bias=True)

    @pytest.fixture
    def quant_config(self):
        return get_int4_quant_config(symmetric=True)

    @pytest.fixture
    def dummy_input(self):
        """Create dummy input tensor."""
        torch.manual_seed(42)
        return torch.randn(4, 128)  # Batch size 4, input dim 128

    def test_forward_quantized_mode(self, dummy_linear, quant_config, dummy_input):
        """Test forward pass in quantized mode (no LoRA)."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Run forward pass
        output = lora_module(dummy_input)

        # Check output shape
        assert output.shape == (4, 256)  # Batch size 4, output dim 256

    def test_forward_materialized_mode(self, dummy_linear, quant_config, dummy_input):
        """Test forward pass with materialized weights (LoRA mode)."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Trigger materialization
        _ = lora_module.weight

        # Set to use materialized weight
        lora_module._use_materialized = True

        # Run forward pass
        output = lora_module(dummy_input)

        # Check output shape
        assert output.shape == (4, 256)

    def test_forward_consistency(self, dummy_linear, quant_config, dummy_input):
        """Test that quantized and materialized modes produce similar outputs."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Run forward in quantized mode
        output_quantized = lora_module(dummy_input)

        # Switch to materialized mode
        _ = lora_module.weight
        lora_module._use_materialized = True
        output_materialized = lora_module(dummy_input)

        # Outputs should be similar (allowing for quantization error)
        # Both decompress the same packed weights
        assert torch.allclose(output_quantized, output_materialized, rtol=1e-5, atol=1e-5)


class TestLoRACompatibleLinearLoRASimulation:
    """Test LoRA adapter simulation."""

    @pytest.fixture
    def dummy_linear(self):
        torch.manual_seed(42)
        return Linear(128, 256, bias=True)

    @pytest.fixture
    def quant_config(self):
        return get_int4_group_quant_config(group_size=128, symmetric=True)

    @pytest.fixture
    def dummy_input(self):
        torch.manual_seed(42)
        return torch.randn(4, 128)

    def test_weight_setter(self, dummy_linear, quant_config, dummy_input):
        """Test that weight setter allows LoRA updates."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Get base weight
        base_weight = lora_module.weight.clone()

        # Simulate LoRA update: add small delta
        lora_delta = torch.randn_like(base_weight) * 0.01
        merged_weight = base_weight + lora_delta

        # Update weight (simulating LoRA injection)
        lora_module.weight = merged_weight

        # Verify weight was updated
        assert torch.allclose(lora_module.weight, merged_weight)

        # Verify module is in materialized mode
        assert lora_module._use_materialized is True

    def test_lora_forward_pass(self, dummy_linear, quant_config, dummy_input):
        """Test forward pass with simulated LoRA adapter."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Run forward with base weights
        output_base = lora_module(dummy_input)

        # Simulate LoRA: weight = base + lora_A @ lora_B
        base_weight = lora_module.weight
        rank = 8
        lora_A = torch.randn(256, rank) * 0.01  # out_features x rank
        lora_B = torch.randn(rank, 128) * 0.01  # rank x in_features
        lora_scale = 1.0

        merged_weight = base_weight + lora_scale * (lora_A @ lora_B)
        lora_module.weight = merged_weight

        # Run forward with LoRA weights
        output_lora = lora_module(dummy_input)

        # Outputs should be different (LoRA changed the weights)
        assert not torch.allclose(output_base, output_lora, rtol=1e-5, atol=1e-5)

        # Output should still be valid
        assert output_lora.shape == (4, 256)
        assert not torch.isnan(output_lora).any()

    def test_clear_materialized_weight(self, dummy_linear, quant_config):
        """Test that clear_materialized_weight frees memory."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Materialize weight
        _ = lora_module.weight
        assert lora_module._materialized_weight is not None

        # Clear materialized weight
        lora_module.clear_materialized_weight()

        # Verify it's cleared
        assert lora_module._materialized_weight is None
        assert lora_module._use_materialized is False

    def test_get_quantization_params(self, dummy_linear, quant_config):
        """Test that quantization parameters are accessible."""
        model = torch.nn.Sequential(dummy_linear)
        apply_quantization_config(model, quant_config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            dummy_linear,
            quantization_scheme=quant_config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Get quantization params
        params = lora_module.get_quantization_params()

        # Verify expected params exist
        assert "quantization_scheme" in params
        assert "weight_scale" in params
        assert "weight_shape" in params

        # Verify scheme is correct
        assert params["quantization_scheme"] == quant_config.config_groups["group_1"]


class TestLoRACompatibleLinearGroupQuantization:
    """Test with group-wise quantization (Kimi K2 configuration)."""

    def test_kimi_k2_config(self):
        """Test with Kimi K2-like configuration: INT4, group_size=128."""
        torch.manual_seed(42)
        linear = Linear(4096, 14336, bias=False)  # Typical MoE expert size

        config = get_int4_group_quant_config(group_size=128, symmetric=True)

        model = torch.nn.Sequential(linear)
        apply_quantization_config(model, config)

        lora_module = LoRACompatibleCompressedLinear.from_linear(
            linear,
            quantization_scheme=config.config_groups["group_1"],
            quantization_format=CompressionFormat.pack_quantized.value,
        )

        # Verify packed weight shape
        # in_features=4096, pack_factor=8 -> packed_size = 4096/8 = 512
        expected_packed_shape = (14336, 512)
        assert lora_module.weight_packed.shape == expected_packed_shape

        # Verify decompressed weight shape
        decompressed_weight = lora_module.weight
        assert decompressed_weight.shape == (14336, 4096)

        # Test forward pass
        input_tensor = torch.randn(2, 4096)
        output = lora_module(input_tensor)
        assert output.shape == (2, 14336)
