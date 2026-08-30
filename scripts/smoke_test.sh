#!/usr/bin/env bash
#
# Execute the notebook labs end to end with reduced settings and report which ones fail.
#
# The labs read their configuration from LAB_* environment variables, so the same notebooks a human
# steps through in Colab can be executed non-interactively here. Every lab asserts its own claims
# with check(...) calls, so "the notebook ran" and "the notebook is correct" are the same statement -
# which is what makes this worth running in CI.
#
# Usage:
#   scripts/smoke_test.sh            # Labs 1 and 2 (no model download, no GPU) - what CI runs
#   scripts/smoke_test.sh all        # every lab, including the ~1 GB Qwen download
#   scripts/smoke_test.sh 3 4        # named labs only
#
# No activated environment is needed: with uv installed the script executes the labs in the
# project environment, syncing the dependency groups the requested labs need.
#
# Written for bash 3.2 so it runs on a stock macOS shell as well as on CI - no associative arrays.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"

# Run from the repository so `uv run` below resolves this project's environment rather than one
# belonging to whatever directory the caller happened to be in. Every path here is absolute, and
# nbconvert executes each notebook with its own copy's directory as the kernel's working
# directory, so nothing else depends on where we were invoked from.
cd "$REPO_ROOT" || exit 1
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

export MPLBACKEND=Agg              # headless plotting
export TOKENIZERS_PARALLELISM=false
export LAB_SEED=0

lab_file() {
  case "$1" in
    1) echo "01_pytorch_lr_mlp.ipynb" ;;
    2) echo "02_transformer_pytorch.ipynb" ;;
    3) echo "03_hf_qwen_lora.ipynb" ;;
    4) echo "04_jax_ray_qwen_lora.ipynb" ;;
    *) echo "" ;;
  esac
}

# Reduced settings, small enough to be quick and large enough that every check still holds. These
# are the values the labs were validated against, not guesses: shrink them further and the accuracy
# assertions start failing on undertraining rather than on a real defect, which is the wrong kind
# of red build.
#
# Two profiles, selected with SMOKE_PROFILE (default "full"):
#
#   full  the validated settings, with margin on every threshold
#   ci    the smallest budget that still passes, for a shared or constrained runner
#
# The "ci" numbers are measured floors, not guesses. Lab 1 at 40/8 epochs keeps real margin
# (linear digits 0.9244 against a 0.90 floor, moons MLP 0.968 against 0.95). Lab 2 stays at 120
# steps in both profiles because it genuinely cannot go lower: at 90 and at 60 steps the
# data-leakage demonstration fails, since the leaky model needs enough training before its
# flattering validation loss actually appears. Cutting it would buy a few seconds and trade a
# real check for a spurious failure.
#
# For reference, on a GitHub ubuntu-latest runner the "ci" profile spends 17s on Lab 1 and 158s on
# Lab 2 out of a ~4 minute job, so execution dominates and Lab 2 is the whole cost. There is no
# knob left to turn there - only a decision about what Lab 2 should demonstrate.
PROFILE="${SMOKE_PROFILE:-full}"
case "$PROFILE" in
  full|ci) ;;
  # Without this a typo would fall through to empty overrides and silently run the labs at their
  # full interactive defaults - minutes of CI time, and a "pass" that tested something else.
  *) echo "unknown SMOKE_PROFILE '$PROFILE' (expected 'full' or 'ci')" >&2; exit 2 ;;
esac

lab_env() {
  case "$PROFILE:$1" in
    full:1) echo "LAB_EPOCHS=60 LAB_DIGIT_EPOCHS=12" ;;
    ci:1)   echo "LAB_EPOCHS=40 LAB_DIGIT_EPOCHS=8" ;;
    *:2)    echo "LAB_MAX_STEPS=120 LAB_EVAL_EVERY=60" ;;
    *:3)    echo "LAB_N_TRAIN=800 LAB_N_EVAL=48 LAB_MAX_STEPS=60 LAB_BATCH_SIZE=4 LAB_GRAD_ACCUM=1" ;;
    *:4)    echo "LAB_N_TRAIN=256 LAB_N_EVAL=32 LAB_MAX_STEPS=30 LAB_BATCH_SIZE=4 LAB_RAY_TRIAL_STEPS=8" ;;
    *)      echo "" ;;
  esac
}

case "${1:-default}" in
  default) LABS="1 2" ;;
  all)     LABS="1 2 3 4" ;;
  *)       LABS="$*" ;;
esac

# Provision the project environment unless one is already activated, so a fresh clone needs `uv
# sync` and nothing else. The groups have to match the labs requested: uv syncs the environment to
# exactly the groups it is given, so a blanket --all-groups would make a Labs 1-2 run (what CI
# does) install transformers, JAX and Ray for nothing, while omitting them would uninstall those
# packages again right before Lab 3 needs them. --frozen keeps this honest: the labs execute what
# uv.lock pins rather than silently relocking.
#
# The labs are then run by putting .venv/bin on PATH - what `activate` does - rather than through
# `uv run`. Ray inspects the driver's command line, and when it finds `uv run` there it tries to
# propagate the uv project to its workers, which fails in Lab 4 because each notebook executes
# from a temporary directory that has no pyproject.toml in it. Activating sidesteps that: Ray sees
# an ordinary virtualenv.
if [ -z "${VIRTUAL_ENV:-}" ] && command -v uv >/dev/null 2>&1; then
  SYNC_GROUPS=""
  case "$LABS" in
    *4*) SYNC_GROUPS="--group jax" ;;   # the jax group includes llm
    *3*) SYNC_GROUPS="--group llm" ;;
  esac

  echo "syncing project environment: uv sync --frozen $SYNC_GROUPS"
  # shellcheck disable=SC2086  # word splitting of SYNC_GROUPS is intentional
  uv sync --frozen $SYNC_GROUPS || exit 1

  VIRTUAL_ENV="$REPO_ROOT/.venv"
  PATH="$VIRTUAL_ENV/bin:$PATH"
  export VIRTUAL_ENV PATH
fi

echo "running labs: $LABS"
FAILED=""

for lab in $LABS; do
  file="$(lab_file "$lab")"
  if [ -z "$file" ]; then
    echo "unknown lab '$lab' (expected 1-4)" >&2
    exit 2
  fi

  env_vars="$(lab_env "$lab")"
  echo
  echo "=============================================================="
  echo "Lab $lab - $file"
  echo "  overrides: $env_vars"
  echo "=============================================================="

  cp "$NOTEBOOK_DIR/$file" "$WORK_DIR/$file"
  start=$SECONDS
  log="$WORK_DIR/lab$lab.log"

  # Capture nbconvert's own exit status rather than a pipeline's. Piping straight into `grep -v`
  # would report a PASS as a failure whenever the filter happens to drop every line, since grep
  # exits 1 when it matches nothing.
  # shellcheck disable=SC2086  # word splitting of env_vars is intentional
  env $env_vars jupyter nbconvert \
      --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 \
      "$WORK_DIR/$file" > "$log" 2>&1
  rc=$?

  grep -vE "FigureCanvasAgg|Kernel is running over TCP" "$log" || true

  if [ $rc -eq 0 ]; then
    echo "Lab $lab PASSED in $((SECONDS - start))s"
  else
    echo "Lab $lab FAILED after $((SECONDS - start))s (exit $rc)" >&2
    FAILED="$FAILED $lab"
  fi
done

echo
if [ -n "$FAILED" ]; then
  echo "FAILED labs:$FAILED" >&2
  exit 1
fi
echo "all labs passed"
