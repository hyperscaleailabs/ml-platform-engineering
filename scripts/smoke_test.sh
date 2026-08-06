#!/usr/bin/env bash
#
# Execute the notebook labs end to end with reduced settings and fail on the first error.
#
# The labs read their configuration from LAB_* environment variables, so the same notebooks that a
# human steps through in Colab can be executed non-interactively here. Every lab asserts its own
# claims with check(...) calls, so "the notebook ran" and "the notebook is correct" are the same
# statement - which is what makes this worth running in CI.
#
# Usage:
#   scripts/smoke_test.sh            # Labs 1 and 2 (no model download, no GPU) - what CI runs
#   scripts/smoke_test.sh all        # every lab, including the ~1 GB Qwen download
#   scripts/smoke_test.sh 3 4        # named labs only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Headless matplotlib, and no tokenizer fork warnings.
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export LAB_SEED=0

declare -A LAB_FILES=(
  [1]=01_pytorch_lr_mlp.ipynb
  [2]=02_transformer_pytorch.ipynb
  [3]=03_hf_qwen_lora.ipynb
  [4]=04_jax_ray_qwen_lora.ipynb
)

# Per-lab overrides: small enough to be quick, large enough that every check still holds.
declare -A LAB_ENV=(
  [1]="LAB_EPOCHS=60 LAB_DIGIT_EPOCHS=12"
  [2]="LAB_MAX_STEPS=120 LAB_EVAL_EVERY=60"
  [3]="LAB_N_TRAIN=400 LAB_N_EVAL=64 LAB_MAX_STEPS=40 LAB_BATCH_SIZE=4 LAB_GRAD_ACCUM=1"
  [4]="LAB_N_TRAIN=256 LAB_N_EVAL=32 LAB_MAX_STEPS=30 LAB_BATCH_SIZE=2 LAB_RAY_TRIAL_STEPS=10"
)

case "${1:-default}" in
  default) LABS=(1 2) ;;
  all)     LABS=(1 2 3 4) ;;
  *)       LABS=("$@") ;;
esac

echo "running labs: ${LABS[*]}"
FAILED=()

for lab in "${LABS[@]}"; do
  file="${LAB_FILES[$lab]:-}"
  if [[ -z "$file" ]]; then
    echo "unknown lab '$lab' (expected 1-4)" >&2
    exit 2
  fi

  echo
  echo "=============================================================="
  echo "Lab $lab - $file"
  echo "  overrides: ${LAB_ENV[$lab]}"
  echo "=============================================================="

  cp "$NOTEBOOK_DIR/$file" "$WORK_DIR/$file"
  start=$SECONDS

  if env ${LAB_ENV[$lab]} jupyter nbconvert \
        --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 \
        "$WORK_DIR/$file" 2>&1 | grep -vE "FigureCanvasAgg|Kernel is running over TCP"; then
    echo "Lab $lab PASSED in $((SECONDS - start))s"
  else
    echo "Lab $lab FAILED after $((SECONDS - start))s" >&2
    FAILED+=("$lab")
  fi
done

echo
if (( ${#FAILED[@]} )); then
  echo "FAILED labs: ${FAILED[*]}" >&2
  exit 1
fi
echo "all labs passed"
