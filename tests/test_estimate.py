"""Tests for the VRAM estimator."""

import pytest

from vramplan.estimate import estimate, fits, max_context
from vramplan.specs import ModelSpec, lookup

SPEC = ModelSpec("Qwen2.5-7B", 7_615_203_328, 28, 3584, 28, 4, 128)


def test_weights_are_dtype_scaled():
    b16 = estimate(SPEC, 4096, weight_dtype="bf16")
    b8 = estimate(SPEC, 4096, weight_dtype="int8")
    # weights dominate and halve when going from 2 to 1 byte/param
    assert b8.weights == pytest.approx(b16.weights / 2, rel=1e-3)


def test_kv_cache_scales_linearly_with_context():
    b1 = estimate(SPEC, 4096)
    b2 = estimate(SPEC, 8192)
    assert b2.kv_cache == pytest.approx(2 * b1.kv_cache, rel=1e-6)


def test_kv_cache_scales_with_batch():
    b1 = estimate(SPEC, 4096, batch_size=1)
    b8 = estimate(SPEC, 4096, batch_size=8)
    assert b8.kv_cache == pytest.approx(8 * b1.kv_cache, rel=1e-6)


def test_known_7b_kv_per_token():
    # Qwen2.5-7B: 28 layers x 4 kv heads x 128 dim x 2 (K+V) x 2 bytes = 57344 B
    b = estimate(SPEC, 1, batch_size=1, weight_dtype="bf16")
    per_token = b.kv_cache * 1024**3  # back to bytes (context_len=1)
    assert per_token == pytest.approx(28 * 4 * 128 * 2 * 2, rel=1e-3)


def test_fits_and_max_context_are_consistent():
    b = estimate(SPEC, 8192)
    assert fits(b, 24.0) == (b.total <= 24.0)
    mc = max_context(SPEC, 48.0, batch_size=1)
    # the estimated max context should fit, and one more block should not
    assert fits(estimate(SPEC, mc), 48.0)
    assert not fits(estimate(SPEC, mc + 4096), 48.0)


def test_lookup_is_case_and_prefix_insensitive():
    assert lookup("Qwen2.5-7B") == lookup("qwen2.5-7b")
    assert lookup("Qwen/Qwen2.5-7B-Instruct") == lookup("Qwen2.5-7B")


def test_lookup_unknown_raises():
    with pytest.raises(KeyError):
        lookup("NoSuchModel-999B")


def test_int4_quantization_halves_weights_vs_int8():
    b8 = estimate(SPEC, 4096, weight_dtype="int8")
    b4 = estimate(SPEC, 4096, weight_dtype="int4")
    assert b4.weights == pytest.approx(b8.weights / 2, rel=1e-3)
