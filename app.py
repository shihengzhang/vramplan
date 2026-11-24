"""Gradio UI for vramplan — interactive GPU memory estimator."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from vramplan import estimate, fits, max_context, lookup
from vramplan.specs import CATALOG

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#f0f0f0",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

COLORS = {"weights": "#5b8def", "KV cache": "#e07b39", "activations": "#2ecc71", "overhead": "#95a5a6"}
MODELS = list(CATALOG)


def _plot(b, gpu_gib):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = ["weights", "KV cache", "activations", "overhead"]
    values = [b.weights, b.kv_cache, b.activations, b.overhead]
    colors = [COLORS[k] for k in labels]
    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(v + gpu_gib * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} GiB", va="center", fontsize=11, color="#333")
    ax.axvline(gpu_gib, color="#e74c3c", ls="--", lw=1.5, zorder=2)
    ax.text(gpu_gib + gpu_gib * 0.01, -0.35, f"GPU {gpu_gib:.0f} GiB", color="#e74c3c", fontsize=10)
    ax.set_xlim(0, max(b.total, gpu_gib) * 1.25)
    ax.invert_yaxis()
    ax.set_xlabel("GiB")
    ax.set_title("显存分解", fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    return fig


def compute(model, context, batch, dtype, kv_dtype, gpu):
    spec = lookup(model)
    kv_dtype = kv_dtype if kv_dtype != "auto" else None
    b = estimate(spec, context, batch, weight_dtype=dtype, kv_dtype=kv_dtype)
    ok = fits(b, gpu)
    mc = max_context(spec, gpu, batch, weight_dtype=dtype, kv_dtype=kv_dtype)
    verdict = "✅ 装得下" if ok else "❌ 装不下"
    summary = (
        f"**总计：{b.total:.2f} GiB**　{verdict}\n\n"
        f"- 权重：{b.weights:.2f} GiB\n"
        f"- KV 缓存：{b.kv_cache:.2f} GiB\n"
        f"- 激活：{b.activations:.2f} GiB\n"
        f"- 开销：{b.overhead:.2f} GiB\n\n"
        f"在 {gpu:.0f} GiB 上，batch {batch} 最大上下文 ≈ **{mc:,} tokens**"
    )
    return _plot(b, gpu), summary


with gr.Blocks(title="vramplan — GPU 显存预估") as demo:
    gr.Markdown(
        "# 🚀 vramplan — GPU 显存预估\n"
        "拖动滑杆，实时看模型需要多少显存、能不能装下。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(MODELS, value="Qwen2.5-7B", label="模型")
            context = gr.Slider(1024, 262144, value=32768, step=1024, label="上下文长度 (tokens)")
            batch = gr.Slider(1, 128, value=4, step=1, label="batch size")
            dtype = gr.Dropdown(["bf16", "fp16", "fp8", "int8", "int4"], value="bf16", label="权重精度")
            kv_dtype = gr.Dropdown(["auto", "bf16", "fp16", "fp8", "int8"], value="auto", label="KV 缓存精度")
            gpu = gr.Slider(8, 512, value=48, step=8, label="GPU 显存 (GiB)")
        with gr.Column(scale=1):
            plot = gr.Plot(label="")
            summary = gr.Markdown()

    inputs = [model, context, batch, dtype, kv_dtype, gpu]
    outputs = [plot, summary]
    for c in inputs:
        c.change(compute, inputs, outputs)
    demo.load(compute, inputs, outputs)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="127.0.0.1", server_port=7860)
