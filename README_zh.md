<div align="center">

# LaPA<sup>2</sup>：面向长文本可控生成的长度感知 Prefix 与 Prompt 注意力增强

**一种长度感知的注意力 logit 偏置，让基于 prefix 的可控文本生成（CTG）在长序列上依然能稳定保持目标属性。**

<p>
  <a href="#"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

[English](./README.md) | **简体中文**

</div>

---

本仓库是论文 *"LaPA<sup>2</sup>: Length-Aware Prefix and Prompt Attention Augmentation for Long-Form Controllable Text Generation"* 的官方代码实现。

LaPA<sup>2</sup> 是一个**无需训练、零额外参数、与模型无关**的、面向基于 prefix 的可控文本生成（CTG）任务的即插即用模块。它针对的核心问题是 *Attention Dilution*（注意力稀释）：随着生成序列长度 `l` 的增加，prefix token 上分配到的注意力会以 `O(l⁻¹)` 的速率衰减，导致控制信号"褪色"。LaPA<sup>2</sup> 在每一个解码步对 prefix 区域的注意力 logit 加一个长度感知的偏置：

```
attn_logit[:, :, :, :boost_len] += alpha * log(kv_seq_len / boost_len)
```

其中 `boost_len` 是属性 token 的长度（软 prefix 长度或 hard prefix 的 token 数量），`alpha > 0` 控制增强强度。可选的 **Contextual Anchor Reinforcement** 把同样的偏置施加到 prompt token 上以保持语义一致性。

LaPA<sup>2</sup> 可以无须微调地插入到一系列 prefix-based CTG 方法上：

| 方法                | 支持的 backbone                            | 文件                       |
| ------------------- | ------------------------------------------ | -------------------------- |
| Prefix-Tuning       | GPT-2（带训练好的 soft prefix）            | `prefix_tuning.py`         |
| Air-Decoding        | GPT-2（带训练好的 soft prefix）            | `air_decoding.py`          |
| PREADD / NegPrompt  | LLaMA-2、Pythia（hard prefix）             | `preadd.py`                |
| Palette of LMs      | LLaMA-2、Pythia（hard prefix）             | `palette.py`               |

论文中使用的四个 backbone 覆盖 ~355M 到 13B 参数规模：**GPT-2 Medium** (355M)、**LLaMA-2-7B**、**LLaMA-2-13B**、**Pythia-12B**。LaPA<sup>2</sup> 的 `set_boost_config(...)` 钩子与任务和方法都无关，要扩展到其他 causal LM 也很直接。

### 🧪 论文中对比的其他 baseline

对于论文中我们额外汇报的 baseline——**Contrastive Prefix**（Qian 等，2022）、**FreeCtrl**（Feng 等，2024）、**DATG**（Liang 等，2024）——我们遵循各自方法的官方实现。如需复现这部分数字，请参考对应论文的官方仓库；本仓库不再重新分发它们的代码或权重。

## 📁 仓库结构

```
LaPA2/
├── air_decoding.py          # Air-Decoding + LaPA2（GPT-2 PC-LMs）
├── prefix_tuning.py         # Prefix-Tuning + LaPA2（GPT-2 PC-LMs）
├── preadd.py                # PREADD / NegPrompt + LaPA2（LLaMA / Pythia）
├── palette.py               # Palette of LMs + LaPA2（LLaMA / Pythia）
├── train_PCLMs.py           # 训练 GPT-2 PC-LMs
├── modeling_gpt2.py         # GPT-2，带 set_boost_config(...) LaPA2 钩子
├── modeling_llama.py        # LLaMA，带 set_boost_config(...) LaPA2 钩子
├── modeling_gpt_neox.py     # GPT-NeoX/Pythia，带 set_boost_config(...) LaPA2 钩子
├── eval_sent_acc.py         # 情感属性准确率评测
├── eval_topic_acc.py        # 主题属性准确率评测
├── eval_toxic.py            # 单样本 Perspective API 毒性评测
├── eval_toxic_batch.py      # 批量 Perspective API 毒性评测
├── eval_perplexity.py       # 用更强的 LM 算 PPL 评流畅度
├── eval_dist.py             # 多样性（Dist-1/2/3）
├── model_sent.py            # 情感分类 RoBERTa 头（评测用）
├── model_topic.py           # 主题分类 RoBERTa 头（评测用）
├── dataset/                 # 三个 CTG 任务的训练/测试数据
├── scripts/                 # 各方法、各任务、各评测的入口 shell 脚本
└── utils/                   # Perspective API 工具等
```

论文中评测的三个 CTG 任务与 Air-Decoding（EMNLP 2023）一致：
* **sentiment**：IMDB prompt，positive / negative 二分类属性
* **topic**：AGNews prompt，world / sports / business / science 四分类
* **detoxification**：Jigsaw prompt，目标是生成非毒性续写

## ⚙️ 环境配置

```bash
conda create -n LaPA2 python=3.10 -y
conda activate LaPA2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

测试环境：`torch==2.2.0+cu121`、`transformers==4.31.0`、CUDA 12.1。
如果你的 CUDA 版本不同，请把 `requirements.txt` 中 `torch==` / `torchvision==` / `torchaudio==` 的 `+cu121` 后缀替换为对应版本（比如 `+cu118`），并同步修改 `--extra-index-url`。

GPT-2 系列的 decoder（Air-Decoding、Prefix-Tuning）需要预训练好的 PC-LM 权重。请从 [Air-Decoding 官方发布](https://drive.google.com/file/d/1Su5-QT2nIjjZ_pcyGkc5f-AR6vOs0ZVw/view?usp=sharing)下载 checkpoint 包（包含 sentiment、topic、detoxification 的 PC-LM 以及对应的 RoBERTa 属性分类器），解压到 `./models/`：

```
./models/
├── ckpt_for_sentiment_and_topic   # 情感 + 主题任务的 PC-LM（air_decoding.py 和 prefix_tuning.py 共用）
├── ckpt_for_detoxification        # 去毒任务的 PC-LM
├── best_sentiment_classifier      # 情感分类器（eval_sent_acc.py 用）
└── best_topic_classifier          # 主题分类器（eval_topic_acc.py 用）
```

如果想从零训练 PC-LM：

```bash
mkdir -p ckpt
bash scripts/train_PCLMs.sh
```

LLaMA-2 / Pythia 的 decoder（`preadd.py`、`palette.py`）会从 HuggingFace Hub 自动下载权重，无需额外 checkpoint。

## 🚀 生成

每个脚本都暴露了 `alpha` 超参数。`alpha=0` 退化为原始 baseline；常用取值在 1~2 之间。

### 🪶 GPT-2 PC-LM 系列 decoder

```bash
bash scripts/air_decoding_sentiment.sh         # Air-Decoding + LaPA2 在 IMDB 上
bash scripts/air_decoding_topic.sh             # Air-Decoding + LaPA2 在 AGNews 上
bash scripts/air_decoding_detoxification.sh    # Air-Decoding + LaPA2 在 Jigsaw 上
bash scripts/prefix_tuning_generate.sh         # Prefix-Tuning + LaPA2
```

### 🦙 大模型 + hard prefix 系列 decoder

```bash
bash scripts/preadd_generate.sh                # PREADD / NegPrompt + LaPA2
bash scripts/palette_generate.sh               # Palette + LaPA2
```

生成结果会写到 `./test_data/<method>/<task>_<config>.jsonl`。

## 📊 评测

```bash
bash scripts/eval_sent_acc.sh    ./test_data/air_decoding/sentiment_140.0_len512_alpha2.jsonl
bash scripts/eval_topic_acc.sh   ./test_data/air_decoding/topic_60.0_len512_alpha2.jsonl
bash scripts/eval_toxic.sh       ./test_data/air_decoding/detoxification_120.0_len50_alpha2.jsonl
bash scripts/eval_perplexity.sh  <jsonl>
bash scripts/eval_dist.sh        <jsonl>
```

`eval_toxic.sh` 会调用 Perspective API，需要把 key 写到环境变量里：

```bash
export PERSPECTIVE_API_KEY=<your key>
bash scripts/eval_toxic.sh <jsonl>
```

## 🔧 LaPA<sup>2</sup> 的接入方式

每个 backbone（`modeling_gpt2.py`、`modeling_llama.py`、`modeling_gpt_neox.py`）都暴露同一个接口：

```python
import modeling_llama
modeling_llama.set_boost_config(enabled=True, alpha=2.0, boost_len=hard_prefix_len)
out = model(...)                          # 这一步的 forward 会带上 LaPA2 boost
modeling_llama.set_boost_config(enabled=False, alpha=0, boost_len=0)
```

decoder 在每次 forward 之前把 boost 长度（属性分支用 `prefix_len`，base 分支启用 anchor 时用 `prompt_len`）写入全局配置、forward 完关掉。boost 只在解码步（query length == 1）触发，prompt forward 阶段是 no-op。

**Contextual Anchor Reinforcement.** 对 Bayesian-reweighting 类 decoder（`air_decoding.py`）来说，anchor 分支的连线方式严格按照论文：当 `--alpha > 0` 时，属性分支 boost *prefix 区域*（soft `prefix_len` 或 hard prefix 的 token 数），base 分支同步 boost *prompt 区域*（`prompt_len`）—— **去毒任务除外**：去毒任务中 base 分支的 anchor 是有意关闭的。这与论文的实验设定以及 Air-Decoding 在 Jigsaw 上的原始 protocol 一致（去毒任务的输入只是 continuation seed，base 分支没有有意义的 prompt 可锚定）。

## 📝 引用

arXiv 版本上线之前，请按如下方式引用：

```bibtex
@unpublished{yang2026lapa2,
  title  = {LaPA\textsuperscript{2}: Length-Aware Prefix and Prompt Attention Augmentation for Long-Form Controllable Text Generation},
  author = {Yang, Jiabing and others},
  year   = {2026}
}
```

<!-- TODO: arXiv preprint 上线后，把上面这条换成对应的 @misc 条目。 -->

本仓库基于 Air-Decoding（Zhong 等，EMNLP 2023）的代码。如果你使用了 GPT-2 PC-LM 的训练 / 解码 pipeline，也请引用原工作：

```bibtex
@inproceedings{zhong-etal-2023-air,
  title    = {Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation},
  author   = {Zhong, Tianqi and Wang, Quan and Han, Jingxuan and Zhang, Yongdong and Mao, Zhendong},
  booktitle= {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year     = {2023},
  url      = {https://aclanthology.org/2023.emnlp-main.512}
}
```

## 📄 协议

本项目沿用原始 Air-Decoding 仓库的 MIT 协议，详见 [LICENSE](LICENSE)。
