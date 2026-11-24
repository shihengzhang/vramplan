# vramplan

> 估算 LLM 推理需要多少显存，以及「能不能装下」。

跑模型前先回答这个问题：**「7B @ 32K、batch 8，一张 48G 卡装得下吗？换成 int4 能省多少？最大能开多长上下文？」** `vramplan` 用几行公式把这些算清楚，不再靠「经验 + 试错」。

## 安装

```bash
pip install -e .            # 核心（零依赖）
pip install -e ".[hf]"     # 可选：从 HuggingFace 拉模型配置
```

## 用法

**命令行：**

```bash
vramplan Qwen2.5-7B --context 32768 --batch 4 --dtype bf16 --gpu 48
```

输出：

```
model       : Qwen2.5-7B  (7.62B params)
context     : 32768 tokens, batch 4, dtype bf16
----------------------------------------------
  weights     : 14.18 GiB
  KV cache    : 7.00 GiB
  activations : 10.50 GiB
  overhead    : 4.17 GiB
  TOTAL       : 35.85 GiB
----------------------------------------------
fits in 48 GiB?  YES
max context (batch 4) on 48 GiB : ~53,445 tokens
```

**Python API：**

```python
from vramplan import estimate, fits, max_context, lookup

spec = lookup("Qwen2.5-7B")
b = estimate(spec, context_len=32768, batch_size=4, weight_dtype="bf16")
print(b.as_dict())            # 各分项明细
print(fits(b, 48))            # 能不能装下
print(max_context(spec, 48, batch_size=4))  # 最大上下文
```

## 估算公式

显存 ≈ **权重 + KV 缓存 + 激活 + 开销**：

| 分项 | 公式 | 说明 |
|---|---|---|
| 权重 | `参数量 × 每参数字节` | bf16/fp16=2B、fp8/int8=1B、int4=0.5B |
| KV 缓存 | `层数 × KV头数 × 头维 × 2 × 上下文 × batch × dtype` | K 和 V 各一份 |
| 激活 | `hidden × 上下文 × batch × dtype × 因子` | 推理逐层释放，峰值只有**一层**的中间量（QKV+注意力+SwiGLU MLP+残差 ≈ 12×hidden） |
| 开销 | `固定 CUDA 上下文 + 10% 安全边际` | 碎片、分配器浪费 |

`activation_factor` 和 `safety_margin` 可调，适配不同引擎（flash attention / CUDA graphs 下激活会更小）。

## 内置模型目录

Qwen2.5（0.5B–72B）、Llama-3.1（8B/70B/405B）。也可传自定义 `ModelSpec`，或 `from_hf(name)` 从 HuggingFace 拉精确配置。

## 许可证

[Apache-2.0](LICENSE)
