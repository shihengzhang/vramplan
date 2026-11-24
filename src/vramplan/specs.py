"""Built-in model catalog and ModelSpec definition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model needed to estimate its VRAM footprint."""

    name: str
    num_params: int          # total parameters (exact count)
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int        # GQA/MQA KV heads
    head_dim: int


# A hand-curated set of common models. Values are close approximations; use
# `ModelSpec` directly (or fetch from HF) when you need exact numbers.
CATALOG: dict[str, ModelSpec] = {
    # --- Qwen2.5 ---
    "Qwen2.5-0.5B": ModelSpec("Qwen2.5-0.5B", 494_032_768, 24, 896, 14, 2, 64),
    "Qwen2.5-1.5B": ModelSpec("Qwen2.5-1.5B", 1_543_697_152, 28, 1536, 12, 2, 128),
    "Qwen2.5-3B": ModelSpec("Qwen2.5-3B", 3_089_633_280, 36, 2048, 16, 2, 128),
    "Qwen2.5-7B": ModelSpec("Qwen2.5-7B", 7_615_203_328, 28, 3584, 28, 4, 128),
    "Qwen2.5-14B": ModelSpec("Qwen2.5-14B", 14_770_840_576, 48, 5120, 40, 8, 128),
    "Qwen2.5-32B": ModelSpec("Qwen2.5-32B", 32_760_318_464, 64, 5120, 40, 8, 128),
    "Qwen2.5-72B": ModelSpec("Qwen2.5-72B", 72_711_856_128, 80, 8192, 64, 8, 128),
    # --- Llama 3.1 ---
    "Llama-3.1-8B": ModelSpec("Llama-3.1-8B", 8_030_261_248, 32, 4096, 32, 8, 128),
    "Llama-3.1-70B": ModelSpec("Llama-3.1-70B", 70_553_016_832, 80, 8192, 64, 8, 128),
    "Llama-3.1-405B": ModelSpec("Llama-3.1-405B", 405_290_913_792, 126, 16384, 128, 8, 128),
}


_SUFFIXES = ("-instruct", "-chat", "-base", "-gguf", "-awq", "-gptq", "-bnb")


def _normalize(name: str) -> str:
    key = name.split("/")[-1].lower()
    for suffix in _SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def lookup(name: str) -> ModelSpec:
    """Look up a model by name (case-insensitive, ignores `/` and -Instruct/-Chat suffixes)."""
    key = _normalize(name)
    for k, v in CATALOG.items():
        if _normalize(k) == key:
            return v
    raise KeyError(
        f"model {name!r} not in catalog. Available: {', '.join(CATALOG)}. "
        "Pass an explicit ModelSpec or add it to CATALOG."
    )


def from_hf(name: str) -> ModelSpec:
    """Build a ModelSpec from a HuggingFace config (requires `transformers`)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(name)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    num_params = int(sum(p.numel() for p in _hf_params(name)))
    return ModelSpec(
        name=name,
        num_params=num_params,
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
    )


def _hf_params(name: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    return model.parameters()
