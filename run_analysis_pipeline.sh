#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LOG_PATH=""
TASK_NAME=""
MODEL_NAME=""
LOGS_ROOT="logs"
CLUSTERING_ROOT="clustering_results"
VIS_ROOT="vis_results"

usage() {
  cat <<'EOF'
Usage:
  ./run_analysis_pipeline.sh --log_path <path> --model_name <model> [options]

Required:
  --log_path <path>       Path to the source log. You can omit .jsonl.
  --model_name <name>     Model subdir to use under clustering_results/ and vis_results/.

Optional:
  --task_name <name>         Task name if it cannot be inferred from the log path.
  --python <bin>             Python executable to use. Default: python
  --logs_root <dir>          Default: logs
  --clustering_root <dir>    Default: clustering_results
  --vis_root <dir>           Default: vis_results
  -h, --help                 Show this help text

Example:
  ./run_analysis_pipeline.sh \
    --log_path logs/xbrl_formula/gpt-5/example.jsonl \
    --model_name gpt-5-mini
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log_path)
      LOG_PATH="${2:-}"
      shift 2
      ;;
    --task_name)
      TASK_NAME="${2:-}"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --logs_root)
      LOGS_ROOT="${2:-}"
      shift 2
      ;;
    --clustering_root)
      CLUSTERING_ROOT="${2:-}"
      shift 2
      ;;
    --vis_root)
      VIS_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_PATH" ]]; then
  echo "--log_path is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "--model_name is required." >&2
  usage >&2
  exit 1
fi

COMMON_ARGS=(
  --log_path "$LOG_PATH"
  --model_name "$MODEL_NAME"
)

if [[ -n "$TASK_NAME" ]]; then
  COMMON_ARGS+=(--task_name "$TASK_NAME")
fi

run_step() {
  echo
  echo "==> $*"
  "$@"
}

# run_step "$PYTHON_BIN" -m rpt.analysis.cluster_failures_and_patches \
#   "${COMMON_ARGS[@]}" \
#   --logs_root "$LOGS_ROOT" \
#   --out "$CLUSTERING_ROOT"

# run_step "$PYTHON_BIN" -m rpt.analysis.cluster_fusion \
#   "${COMMON_ARGS[@]}" \
#   --clustering_root "$CLUSTERING_ROOT" \
#   --domain failures

# run_step "$PYTHON_BIN" -m rpt.analysis.cluster_fusion \
#   "${COMMON_ARGS[@]}" \
#   --clustering_root "$CLUSTERING_ROOT" \
#   --domain patches

# run_step "$PYTHON_BIN" -m rpt.analysis.performance_summarization_and_analysis \
#   "${COMMON_ARGS[@]}" \
#   --logs_root "$LOGS_ROOT" \
#   --clustering_root "$CLUSTERING_ROOT" \
#   --vis_root "$VIS_ROOT"

run_step "$PYTHON_BIN" -m rpt.analysis.interpret_data_using_heatmaps \
  "${COMMON_ARGS[@]}" \
  --logs_root "$LOGS_ROOT" \
  --clustering_root "$CLUSTERING_ROOT" \
  --vis_root "$VIS_ROOT"

echo
echo "Analysis pipeline completed."
