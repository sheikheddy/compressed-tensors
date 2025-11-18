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

from typing import Optional

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import QuantizationScheme, QuantizationStatus
from compressed_tensors.utils import register_offload_parameter
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear


class LoRACompatibleCompressedLinear(CompressedLinear):
    """
    Extended CompressedLinear that materializes float weights alongside packed buffers
    for LoRA compatibility.

    This class enables LoRA adapters to work with compressed INT4/INT8 models by:
    1. Exposing a `weight` property that returns materialized float tensors
    2. Lazy decompression: weights are only materialized when first accessed
    3. Maintaining packed buffers for efficient storage
    4. Supporting quantized forward passes or float forward passes based on mode

    Key differences from CompressedLinear:
    - CompressedLinear: Decompresses once on first forward, then uses float weights
    - LoRACompatibleCompressedLinear: Keeps packed weights, materializes on-demand

    Usage for vLLM with LoRA:
        module = LoRACompatibleCompressedLinear.from_linear(
            linear_module,
            quantization_scheme,
            quantization_format
        )

        # vLLM can now access module.weight for LoRA injection
        base_weight = module.weight  # Returns materialized float tensor

        # Apply LoRA delta
        merged_weight = base_weight + lora_A @ lora_B * scale

        # Re-quantize if needed for efficient inference
        module.update_compressed_weight(merged_weight)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._materialized_weight: Optional[Tensor] = None
        self._use_materialized = False

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        module: Linear,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
        materialize_on_init: bool = False,
    ):
        """
        Convert a Linear module to LoRACompatibleCompressedLinear.

        :param module: dense linear module to replace
        :param quantization_scheme: quantization config for the module to wrap
        :param quantization_format: compression format module is stored as
        :param materialize_on_init: if True, materialize weights immediately;
            if False, use lazy materialization (default)
        :return: LoRACompatibleCompressedLinear module wrapping the input module
        """
        # First convert to CompressedLinear using parent method
        compressed_module = CompressedLinear.from_linear(
            module, quantization_scheme, quantization_format
        )

        # Then upgrade to LoRACompatibleCompressedLinear
        compressed_module.__class__ = LoRACompatibleCompressedLinear
        compressed_module._materialized_weight = None
        compressed_module._use_materialized = False

        # Optionally materialize immediately
        if materialize_on_init:
            _ = compressed_module.weight  # Trigger materialization

        # Update forward hook if present
        if hasattr(compressed_module, "_old_forward"):
            compressed_module._old_forward = (
                LoRACompatibleCompressedLinear.forward.__get__(
                    compressed_module, LoRACompatibleCompressedLinear
                )
            )

        return compressed_module

    @property
    def weight(self) -> Tensor:
        """
        Returns materialized float weight for LoRA injection.

        On first access:
        1. Decompresses packed buffers (int32 -> int8 -> float)
        2. Caches the result in _materialized_weight
        3. Returns the float tensor

        On subsequent accesses:
        - Returns cached _materialized_weight

        This property allows vLLM's LoRA injection code to access weights
        as if they were standard float tensors, while maintaining efficient
        packed storage internally.

        :return: Decompressed weight tensor in float format (BF16/FP32)
        """
        if self._materialized_weight is None:
            # Lazy decompression on first access
            if self.quantization_status == QuantizationStatus.COMPRESSED:
                weight_data = self.compressor.decompress_module(self)
                self._materialized_weight = weight_data
            else:
                # Weight was already decompressed by parent class
                # This shouldn't happen in normal usage, but handle it gracefully
                if hasattr(self, "_parameters") and "weight" in self._parameters:
                    self._materialized_weight = self._parameters["weight"]
                else:
                    raise RuntimeError(
                        "Cannot materialize weight: module is not in COMPRESSED state "
                        "and no weight parameter found"
                    )

        return self._materialized_weight

    @weight.setter
    def weight(self, value: Tensor):
        """
        Setter for weight property to support LoRA adapter updates.

        When vLLM applies LoRA deltas, it may update the weight:
            module.weight = base_weight + lora_delta

        This setter:
        1. Stores the new weight in _materialized_weight
        2. Marks the module to use materialized weights in forward pass
        3. Optionally re-quantizes for efficient inference (future enhancement)

        :param value: New weight tensor (typically base + LoRA delta)
        """
        self._materialized_weight = value
        self._use_materialized = True

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with LoRA compatibility.

        Two modes of operation:
        1. Quantized mode (_use_materialized=False):
           - Decompress on-the-fly from packed buffers
           - More memory efficient
           - Used when no LoRA adapters are active

        2. Materialized mode (_use_materialized=True):
           - Use cached materialized weight (with LoRA deltas applied)
           - Less memory efficient but supports LoRA
           - Used after LoRA adapter injection

        :param input: Input tensor
        :return: Output of linear layer
        """
        if self._use_materialized and self._materialized_weight is not None:
            # Use materialized weight (includes LoRA deltas)
            return linear(input, self._materialized_weight, self.bias)
        else:
            # Decompress from packed buffers (no LoRA)
            if self.quantization_status == QuantizationStatus.COMPRESSED:
                # Decompress but don't permanently replace packed weights
                weight_data = self.compressor.decompress_module(self)
                return linear(input, weight_data, self.bias)
            else:
                # Already decompressed by parent class (fallback path)
                if hasattr(self, "_parameters") and "weight" in self._parameters:
                    return linear(input, self._parameters["weight"], self.bias)
                else:
                    # Use materialized weight as fallback
                    return linear(input, self.weight, self.bias)

    def update_compressed_weight(self, new_weight: Tensor):
        """
        Update the compressed weight after LoRA adapter changes.

        This method allows re-quantizing the merged weight (base + LoRA delta)
        back to packed format for efficient inference. This is useful for:
        - Persistent LoRA merges (user wants to keep adapter)
        - Batch inference with same adapter

        Steps:
        1. Quantizes the new weight to int8
        2. Packs to int32 format
        3. Updates the packed weight buffers
        4. Clears materialized weight cache

        Note: This is a future enhancement. Current implementation stores
        the new weight and marks for re-quantization on next compression.

        :param new_weight: Merged weight tensor (base + LoRA delta)
        """
        # Store the new weight
        self._materialized_weight = new_weight
        self._use_materialized = True

        # TODO: Implement re-quantization logic
        # This requires access to quantization parameters (scale, zero_point)
        # and calling the compress_weight method
        #
        # Pseudocode:
        # compressed_data = self.compressor.compress_weight(
        #     weight=new_weight,
        #     scale=self.weight_scale,
        #     quantization_args=self.quantization_scheme.weights,
        #     zero_point=getattr(self, 'weight_zero_point', None),
        #     g_idx=getattr(self, 'weight_g_idx', None),
        # )
        # for key, value in compressed_data.items():
        #     setattr(self, key, Parameter(value, requires_grad=False))
        #
        # self._materialized_weight = None
        # self._use_materialized = False

    def clear_materialized_weight(self):
        """
        Clear the cached materialized weight to free memory.

        Useful when:
        - Switching between LoRA adapters
        - No longer need LoRA compatibility
        - Want to reduce memory usage

        After calling this, next access to .weight will trigger decompression.
        """
        self._materialized_weight = None
        self._use_materialized = False

    def get_quantization_params(self):
        """
        Get quantization parameters for LoRA scaling.

        vLLM or other LoRA implementations may need to know:
        - weight_scale: scaling factors for dequantization
        - weight_zero_point: zero points for asymmetric quantization
        - group_size: granularity of group-wise quantization

        :return: Dictionary of quantization parameters
        """
        params = {
            "quantization_scheme": self.quantization_scheme,
        }

        # Add compression parameters if they exist
        if hasattr(self, "weight_scale"):
            params["weight_scale"] = self.weight_scale
        if hasattr(self, "weight_zero_point"):
            params["weight_zero_point"] = self.weight_zero_point
        if hasattr(self, "weight_g_idx"):
            params["weight_g_idx"] = self.weight_g_idx
        if hasattr(self, "weight_shape"):
            params["weight_shape"] = self.weight_shape

        return params
