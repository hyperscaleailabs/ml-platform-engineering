# ml-platform-engineering

Machine learning platform engineering: infrastructure, tooling, and best practices for building and operating ML systems at scale.

## Overview

This repository is a home for ML platform engineering work - the systems and workflows that take machine learning from notebooks to reliable, production-grade services. Topics of interest include:

- **Training infrastructure** - distributed training, job orchestration, and compute scheduling
- **Model serving** - low-latency inference, batching, and autoscaling
- **Data & feature pipelines** - reproducible datasets and feature stores
- **Experiment tracking** - versioning of data, code, and models
- **CI/CD for ML** - automated evaluation, testing, and deployment
- **Observability** - monitoring, drift detection, and cost tracking

## Notebook Labs

A four-part, hands-on series that starts at tensors and autograd and ends with a fine-tuned language
model running across a compute fleet. Every lab runs unmodified in **Google Colab** or locally, and
verifies its own claims with inline assertions instead of asking you to trust a plot.

| # | Lab | Stack | Runtime on CPU | GPU |
|---|---|---|---|---|
| 1 | [PyTorch Foundations: Logistic Regression and an MLP Classifier](notebooks/01_pytorch_lr_mlp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/01_pytorch_lr_mlp.ipynb) | PyTorch, scikit-learn | ~3 min | not needed |
| 2 | [Transformers from Scratch](notebooks/02_transformer_pytorch.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/02_transformer_pytorch.ipynb) | PyTorch | ~22 min | optional |
| 3 | [Hugging Face: Qwen + LoRA](notebooks/03_hf_qwen_lora.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/03_hf_qwen_lora.ipynb) | transformers, PEFT, datasets | ~20 min | recommended |
| 4 | [JAX and Ray: Qwen + LoRA at platform scale](notebooks/04_jax_ray_qwen_lora.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/04_jax_ray_qwen_lora.ipynb) | JAX, Optax, Ray | ~15 min | recommended |

Runtimes are measured on an Apple M4 CPU at default settings; a Colab T4 GPU is several times
faster for Labs 2 through 4. Every lab is parameterized by environment variables, so a quick pass
is a matter of `LAB_MAX_STEPS=…`.

The labs are cumulative and share one concrete model. Lab 2 implements the exact architecture -
RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings - that Lab 3 fine-tunes as
Qwen2.5-0.5B-Instruct and Lab 4 reimplements in JAX, verified logit-for-logit against the
HuggingFace weights.

The thread running through the series is a working habit rather than a syllabus: **build the
component, then prove it is correct against something independent.** Autograd is checked against a
hand-derived gradient, attention against PyTorch's fused kernel, LoRA against PEFT, the JAX port
against HuggingFace, and a sharded evaluation against its single-process result.

See [notebooks/README.md](notebooks/README.md) for local setup, configuration, and how the labs are
executed in CI.

## Getting Started

```bash
git clone https://github.com/hyperscaleailabs/ml-platform-engineering.git
cd ml-platform-engineering

python -m venv .venv && source .venv/bin/activate
pip install -r notebooks/requirements-core.txt

jupyter lab notebooks/
```

Or open Lab 1 directly in Colab with the badge above - no installation at all.

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
