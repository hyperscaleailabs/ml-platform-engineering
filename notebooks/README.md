# Notebook Labs

Four self-contained labs that build up from tensors and autograd to fine-tuning a real language
model and running the workload across a compute fleet. Each one runs unmodified in **Google Colab**
or **locally**, and every lab verifies its own claims with inline `check(...)` assertions rather
than asking you to eyeball a plot.

| # | Lab | Stack | Runtime on CPU | GPU |
|---|---|---|---|---|
| 1 | [PyTorch Foundations: Logistic Regression and an MLP Classifier](01_pytorch_lr_mlp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/01_pytorch_lr_mlp.ipynb) | PyTorch, scikit-learn | ~3 min | not needed |
| 2 | [Transformers from Scratch](02_transformer_pytorch.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/02_transformer_pytorch.ipynb) | PyTorch | ~22 min | optional |
| 3 | [Hugging Face: Qwen + LoRA](03_hf_qwen_lora.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/03_hf_qwen_lora.ipynb) | transformers, PEFT, datasets | ~20 min | recommended |
| 4 | [JAX and Ray: Qwen + LoRA at platform scale](04_jax_ray_qwen_lora.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyperscaleailabs/ml-platform-engineering/blob/main/notebooks/04_jax_ray_qwen_lora.ipynb) | JAX, Optax, Ray | ~15 min | recommended |

Runtimes are **measured** on an Apple M4 CPU at each lab's default settings. A Colab T4 GPU is
several times faster for Labs 2 through 4. Each lab's header names an environment variable
combination that gives a much quicker pass with every assertion still holding.

The labs are cumulative. Lab 2 builds the exact architecture (RMSNorm, RoPE, grouped-query
attention, SwiGLU, tied embeddings) that Labs 3 and 4 then fine-tune and reimplement, using
Qwen2.5-0.5B-Instruct as the concrete model throughout.

## Running in Colab

Click any badge above. Nothing to install for Labs 1 and 2; Labs 3 and 4 install their extra
dependencies in the first cell. For Labs 3 and 4 pick **Runtime -> Change runtime type -> T4 GPU**
before running.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.10 - 3.12

pip install -r notebooks/requirements-core.txt       # Labs 1 and 2
pip install -r notebooks/requirements-llm.txt        # Lab 3
pip install -r notebooks/requirements-jax.txt        # Lab 4

jupyter lab notebooks/
```

Labs 3 and 4 download Qwen2.5-0.5B-Instruct (~1 GB) and the `dair-ai/emotion` dataset (~1 MB) from
the Hugging Face Hub on first run, and cache them under `~/.cache/huggingface`. Both labs fall back
to a small embedded dataset if the download fails, so they still run offline once the model is
cached.

Everything works on CPU (Labs 3 and 4 are just slower - see the runtime note in each notebook's
setup cell). Apple silicon uses the MPS backend automatically in Labs 1 through 3; Lab 4 runs on
JAX's CPU backend, since JAX has no Metal support.

## Configuration

Every lab reads its settings from environment variables with sensible defaults, so the notebooks can
be executed non-interactively - in CI, or just faster on a slow machine:

```bash
LAB_MAX_STEPS=20 LAB_N_EVAL=32 jupyter nbconvert --to notebook --execute \
  --inplace notebooks/03_hf_qwen_lora.ipynb
```

Common knobs (each notebook lists its own in the config block at the top):

| Variable | Meaning |
|---|---|
| `LAB_SEED` | Seed for NumPy, PyTorch and JAX |
| `LAB_MAX_STEPS` / `LAB_EPOCHS` | Training length |
| `LAB_N_TRAIN` / `LAB_N_EVAL` | Dataset subset sizes (Labs 3 and 4) |
| `LAB_BATCH_SIZE` | Training batch size |
| `LAB_MODEL_ID` | Any Qwen2/Qwen3 checkpoint, e.g. `Qwen/Qwen3-0.6B` (Labs 3 and 4) |
| `LAB_DEVICE` | Force `cuda`, `mps` or `cpu` in Lab 3 |
| `LAB_SKIP_RAY` | Set to `1` to skip Lab 4's Ray section |
| `LAB_RAY_WORKERS` | Parallel trials in Lab 4's sweep (default 2) |

### Two hardware notes worth reading before you start

**Apple silicon, Lab 3.** Labs 1 and 2 use the MPS backend automatically and are fine. Lab 3 does
**not**: running a 0.5B model in fp32 with long prompts through MPS reliably produced
`command buffer exited with error status / Internal Error` from the Metal driver during development
- a driver-level failure PyTorch cannot catch or retry. Lab 3 therefore defaults to CPU on a Mac.
If your setup handles it, opt back in with `LAB_DEVICE=mps`.

**Memory, Lab 4.** The Ray section keeps one shared copy of the base weights in the object store
(~2 GB) plus roughly 2 GB of private JAX memory per worker, so budget about 8 GB at the default of
two workers. Per-process RSS looks higher because the shared mapping is counted in every process
that maps it - which is exactly the effect the object store exists to produce. `LAB_SKIP_RAY=1`
skips the section.

## Notebooks are shipped with outputs cleared

You get a clean diff on every commit and you run the code yourself rather than reading someone
else's cached results. `scripts/smoke_test.sh` executes Labs 1 and 2 with reduced settings and is
what CI runs on every push.

## A note on the `check(...)` calls

Each lab defines:

```python
def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"CHECK FAILED: {message}")
    print(f"  ok - {message}")
```

They are not decoration. They assert things like *autograd matches the hand-derived gradient*,
*attention is strictly causal*, *the JAX port matches HuggingFace's logits*, and *the sharded
evaluation reproduces the single-process result*. If a cell's claim is wrong, the notebook stops
there rather than printing a plausible-looking plot.
