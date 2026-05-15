<div align="center">

# LaPA<sup>2</sup>: Length-Aware Prefix and Prompt Attention Augmentation

**Length-aware logit bias that keeps prefix-based Controllable Text Generation on-attribute across long contexts.**

<p>
  <a href="#"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

**English** | [简体中文](./README_zh.md)

</div>

---

Official code release for our paper, *"LaPA<sup>2</sup>: Length-Aware Prefix and Prompt Attention Augmentation for Long-Form Controllable Text Generation"*.

LaPA<sup>2</sup> is a **training-free, zero-parameter, model-agnostic** add-on for prefix-based Controllable Text Generation (CTG). It mitigates *Attention Dilution* — the empirical observation that prefix attention decays at a rate of `O(l⁻¹)` as the generated sequence length `l` grows — by adding a length-aware bias to the prefix-region attention logits at every decoding step:

```
attn_logit[:, :, :, :boost_len] += alpha * log(kv_seq_len / boost_len)
```

where `boost_len` is the number of attribute tokens (soft prefix length or hard-prefix token count) and `alpha > 0` controls the augmentation strength. An optional **Contextual Anchor Reinforcement** applies the same bias to prompt tokens.

LaPA<sup>2</sup> drops into a wide range of prefix-based CTG methods without any fine-tuning:

| Method                | Backbone(s)                                 | File                       |
| --------------------- | ------------------------------------------- | -------------------------- |
| Prefix-Tuning         | GPT-2 (with trained soft prefixes)          | `prefix_tuning.py`         |
| Air-Decoding          | GPT-2 (with trained soft prefixes)          | `air_decoding.py`          |
| PREADD / NegPrompt    | LLaMA-2, Pythia (hard prefix)               | `preadd.py`                |
| Palette of LMs        | LLaMA-2, Pythia (hard prefix)               | `palette.py`               |

The four backbones used in the paper span ~355M to 13B parameters: **GPT-2 Medium** (355M), **LLaMA-2-7B**, **LLaMA-2-13B**, and **Pythia-12B**. LaPA<sup>2</sup>'s `set_boost_config(...)` hook is task- and method-agnostic, so applying it to other causal LM architectures is straightforward.

### 🧪 Other baselines compared in the paper

For the additional baselines we report against — **Contrastive Prefix** (Qian et al., 2022), **FreeCtrl** (Feng et al., 2024), and **DATG** (Liang et al., 2024) — we follow each method's official implementation. Please refer to the original repositories to reproduce those numbers; we do not redistribute their code or trained weights here.

## 📁 Repository layout

```
LaPA2/
├── air_decoding.py          # Air-Decoding + LaPA2 (GPT-2 PC-LMs)
├── prefix_tuning.py         # Prefix-Tuning + LaPA2 (GPT-2 PC-LMs)
├── preadd.py                # PREADD / NegPrompt + LaPA2 (LLaMA / Pythia)
├── palette.py               # Palette of LMs + LaPA2 (LLaMA / Pythia)
├── train_PCLMs.py           # Train the GPT-2 PC-LMs used by the three GPT-2 decoders
├── modeling_gpt2.py         # GPT-2 with `set_boost_config(...)` LaPA2 hook
├── modeling_llama.py        # LLaMA with `set_boost_config(...)` LaPA2 hook
├── modeling_gpt_neox.py     # GPT-NeoX/Pythia with `set_boost_config(...)` LaPA2 hook
├── eval_sent_acc.py         # Sentiment accuracy on attribute classifier
├── eval_topic_acc.py        # Topic accuracy on attribute classifier
├── eval_toxic.py            # Per-sample toxicity via Perspective API
├── eval_toxic_batch.py      # Batched Perspective API toxicity scoring
├── eval_perplexity.py       # Fluency via stronger LM perplexity
├── eval_dist.py             # Diversity (Dist-1/2/3)
├── model_sent.py            # RoBERTa sentiment classifier head (eval-only)
├── model_topic.py           # RoBERTa topic classifier head (eval-only)
├── dataset/                 # Training/test data for the three CTG tasks
├── scripts/                 # Runnable bash entry points
└── utils/                   # Perspective API helpers, etc.
```

The three CTG tasks follow Air-Decoding (EMNLP 2023):
* **sentiment** — IMDB prompts, positive/negative target attributes
* **topic** — AGNews prompts, four-way world/sports/business/science
* **detoxification** — Jigsaw prompts; we generate continuations that should be non-toxic

## ⚙️ Setup

```bash
conda create -n LaPA2 python=3.10 -y
conda activate LaPA2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Tested with `torch==2.2.0+cu121`, `transformers==4.31.0`, CUDA 12.1.
If you are on a different CUDA version, edit the `torch==`/`torchvision==`/`torchaudio==`
pins in `requirements.txt` (e.g. swap `+cu121` for `+cu118`) and adjust the
`--extra-index-url` accordingly.

The GPT-2 decoders (Air-Decoding, Prefix-Tuning) need pretrained PC-LM weights. Download the original Air-Decoding checkpoint bundle (sentiment, topic, detoxification PC-LMs plus the RoBERTa attribute classifiers) from the [Air-Decoding release](https://drive.google.com/file/d/1Su5-QT2nIjjZ_pcyGkc5f-AR6vOs0ZVw/view?usp=sharing) and unpack it into `./models/`:

```
./models/
├── ckpt_for_sentiment_and_topic   # PC-LM for sentiment + topic (used by air_decoding.py and prefix_tuning.py)
├── ckpt_for_detoxification        # PC-LM for detoxification
├── best_sentiment_classifier      # RoBERTa classifier for eval_sent_acc.py
└── best_topic_classifier          # RoBERTa classifier for eval_topic_acc.py
```

If you would rather train the PC-LMs from scratch:

```bash
mkdir -p ckpt
bash scripts/train_PCLMs.sh
```

The LLaMA-2 / Pythia decoders (`preadd.py`, `palette.py`) download their base weights on the fly from the HuggingFace Hub — no extra checkpoint required.

## 🚀 Generation

Each script exposes the `alpha` hyperparameter. Setting `alpha=0` recovers the original (vanilla) baseline; values around 1 or 2 are reasonable defaults for LaPA<sup>2</sup>.

### 🪶 GPT-2 PC-LM decoders

```bash
bash scripts/air_decoding_sentiment.sh         # Air-Decoding + LaPA2 on IMDB
bash scripts/air_decoding_topic.sh             # Air-Decoding + LaPA2 on AGNews
bash scripts/air_decoding_detoxification.sh    # Air-Decoding + LaPA2 on Jigsaw
bash scripts/prefix_tuning_generate.sh         # Prefix-Tuning + LaPA2
```

### 🦙 Large-model hard-prefix decoders

```bash
bash scripts/preadd_generate.sh                # PREADD / NegPrompt + LaPA2
bash scripts/palette_generate.sh               # Palette + LaPA2
```

Generated outputs are written to `./test_data/<method>/<task>_<config>.jsonl`.

## 📊 Evaluation

```bash
bash scripts/eval_sent_acc.sh    ./test_data/air_decoding/sentiment_140.0_len512_alpha2.jsonl
bash scripts/eval_topic_acc.sh   ./test_data/air_decoding/topic_60.0_len512_alpha2.jsonl
bash scripts/eval_toxic.sh       ./test_data/air_decoding/detoxification_120.0_len50_alpha2.jsonl
bash scripts/eval_perplexity.sh  <jsonl>
bash scripts/eval_dist.sh        <jsonl>
```

`eval_toxic.sh` queries the Perspective API and expects the key in `PERSPECTIVE_API_KEY`:

```bash
export PERSPECTIVE_API_KEY=<your key>
bash scripts/eval_toxic.sh <jsonl>
```

## 🔧 How LaPA<sup>2</sup> is wired into the models

Each backbone (`modeling_gpt2.py`, `modeling_llama.py`, `modeling_gpt_neox.py`) exposes the same one-function API:

```python
import modeling_llama
modeling_llama.set_boost_config(enabled=True, alpha=2.0, boost_len=hard_prefix_len)
out = model(...)                         # one forward step with LaPA2 on prefix tokens
modeling_llama.set_boost_config(enabled=False, alpha=0, boost_len=0)
```

Decoders set the boost length per branch and toggle the flag around each forward call. The boost is applied only at decoding steps (query length `1`), so it is a no-op when running the prompt forward pass.

**Contextual Anchor Reinforcement.** For the Bayesian-reweighting decoder (`air_decoding.py`), LaPA<sup>2</sup>'s anchor branch is wired exactly as described in the paper: when `--alpha > 0`, the attribute branch boosts the *prefix region* (soft `prefix_len` or hard-prefix token count), and the base branch synchronously boosts the *prompt region* (`prompt_len`) — **except for the detoxification task**, where the base-branch anchor is intentionally disabled. This matches the paper's experimental setup and the original Air-Decoding-on-Jigsaw protocol (the base branch has no informative prompt to anchor on, since the input is just a continuation seed).

## 📝 Citing

A preprint will be released on arXiv. Until the arXiv ID is available, please cite as:

```bibtex
@unpublished{yang2026lapa2,
  title  = {LaPA\textsuperscript{2}: Length-Aware Prefix and Prompt Attention Augmentation for Long-Form Controllable Text Generation},
  author = {Yang, Jiabing and others},
  year   = {2026}
}
```

<!-- TODO: replace with @misc arXiv entry once the preprint is posted. -->

The codebase builds on Air-Decoding (Zhong et al., EMNLP 2023); please also cite the original work if you use the GPT-2 PC-LM training and decoding pipeline:

```bibtex
@inproceedings{zhong-etal-2023-air,
  title    = {Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation},
  author   = {Zhong, Tianqi and Wang, Quan and Han, Jingxuan and Zhang, Yongdong and Mao, Zhendong},
  booktitle= {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year     = {2023},
  url      = {https://aclanthology.org/2023.emnlp-main.512}
}
```

## 📄 License

This project inherits the original Air-Decoding license (see `LICENSE`).
