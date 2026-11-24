"""VRAM estimation for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass

from .specs import ModelSpec

# Bytes per parameter / per element for common dtypes.
DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2, "bf16": 2,
    "fp8": 1, "int8": 1,
    "int4": 0.5,
}

GIB = 1024**3


@dataclass
class MemoryBreakdown:
    """VRAM breakdown in GiB."""

    weights: float
    kv_cache: float
    activations: float
    overhead: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "weights_gib": round(self.weights, 2),
            "kv_cache_gib": round(self.kv_cache, 2),
            "activations_gib": round(self.activations, 2),
            "overhead_gib": round(self.overhead, 2),
            "total_gib": round(self.total, 2),
        }


def estimate(
    spec: ModelSpec,
    context_len: int,
    batch_size: int = 1,
    weight_dtype: str = "bf16",
    kv_dtype: str | None = None,
    activation_factor: float = 12.0,
    fixed_overhead_gib: float = 1.0,
    safety_margin: float = 0.10,
) -> MemoryBreakdown:
    """Estimate the peak VRAM for serving ``spec`` at a given context length.

    The estimate decomposes into four terms (see the README for the derivation):

    * ``weights``      — params x bytes-per-param (dtype/quantization)
    * ``kv_cache``     — layers x kv_heads x head_dim x 2 x seq x batch
    * ``activations``  — hidden x seq x batch x dtype x factor (one layer's peak;
      inference frees each layer's activations as it goes, so the peak is a single
      layer's intermediates, not all layers at once)
    * ``overhead``     — fixed CUDA context + a safety margin on the rest

    ``activation_factor`` approximates the per-layer transient tensors (QKV,
    attention output, SwiGLU up/gate/down, residuals, norms ≈ 12x hidden).
    ``safety_margin`` adds headroom for fragmentation/allocator waste.
    """
    wd = DTYPE_BYTES.get(weight_dtype, 2)
    kd = DTYPE_BYTES.get(kv_dtype or weight_dtype, wd)

    weights = spec.num_params * wd
    kv_per_token = spec.num_layers * spec.num_kv_heads * spec.head_dim * 2 * kd
    kv_cache = kv_per_token * context_len * batch_size
    activations = spec.hidden_size * context_len * batch_size * wd * activation_factor

    overhead = fixed_overhead_gib * GIB + (weights + kv_cache + activations) * safety_margin

    return MemoryBreakdown(
        weights=weights / GIB,
        kv_cache=kv_cache / GIB,
        activations=activations / GIB,
        overhead=overhead / GIB,
        total=(weights + kv_cache + activations + overhead) / GIB,
    )


def fits(breakdown: MemoryBreakdown, gpu_memory_gib: float) -> bool:
    """Whether the estimate fits in a GPU with ``gpu_memory_gib`` VRAM."""
    return breakdown.total <= gpu_memory_gib


def max_context(
    spec: ModelSpec,
    gpu_memory_gib: float,
    batch_size: int = 1,
    weight_dtype: str = "bf16",
    kv_dtype: str | None = None,
    activation_factor: float = 12.0,
    fixed_overhead_gib: float = 1.0,
    safety_margin: float = 0.10,
) -> int:
    """Largest context length that fits in ``gpu_memory_gib`` (solve for seq).

    Solves ``weights + (kv + act) * seq + overhead = budget`` for seq, where the
    per-token ``kv + act`` is computed from the model specs and dtypes.
    """
    wd = DTYPE_BYTES.get(weight_dtype, 2)
    kd = DTYPE_BYTES.get(kv_dtype or weight_dtype, wd)

    weights = spec.num_params * wd
    kv_per_token = spec.num_layers * spec.num_kv_heads * spec.head_dim * 2 * kd
    act_per_token = spec.hidden_size * wd * activation_factor

    # budget for the seq-dependent part, after weights + fixed overhead + margin
    budget = gpu_memory_gib * GIB - fixed_overhead_gib * GIB
    per_token = (kv_per_token + act_per_token) * batch_size
    seq = (budget - weights * (1 + safety_margin)) / (per_token * (1 + safety_margin))
    return max(0, int(seq))
