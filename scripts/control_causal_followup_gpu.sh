#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <start|status|stop|kill|tail> <profile> [functional|profiling|full]"
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ACTION="$1"
PROFILE="$2"
MODE="${3:-}"
[[ "$PROFILE" =~ ^(qwen|phi|mistral)$ ]] || { echo "invalid profile" >&2; exit 2; }

STATE_DIR="${CAUSAL_STATE_DIR:-.operator/causal}"
mkdir -p "$STATE_DIR"
PID_FILE="$STATE_DIR/${PROFILE}.pid"
MODE_FILE="$STATE_DIR/${PROFILE}.mode"
if [[ -z "$MODE" && -f "$MODE_FILE" ]]; then
  MODE="$(cat "$MODE_FILE")"
fi
MODE="${MODE:-full}"
[[ "$MODE" =~ ^(functional|profiling|full)$ ]] || { echo "invalid mode" >&2; exit 2; }
LOG_FILE="$STATE_DIR/${PROFILE}-${MODE}.log"
LOCK_FILE="$STATE_DIR/${PROFILE}.lock"

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

case "$ACTION" in
  start)
    if pid="$(running_pid)"; then
      echo "already running pid=$pid" >&2
      exit 3
    fi
    : > "$LOG_FILE"
    echo "$MODE" > "$MODE_FILE"
    nohup bash -c '
      set -euo pipefail
      lock_file="$1"; profile="$2"; mode="$3"; log_file="$4"
      exec 9>"$lock_file"
      flock -n 9 || { echo "profile lock is held" >&2; exit 73; }
      exec scripts/run_causal_followup_gpu.sh "$profile" "$mode"
    ' bash "$LOCK_FILE" "$PROFILE" "$MODE" "$LOG_FILE" >>"$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"
    echo "started profile=$PROFILE mode=$MODE pid=$pid log=$LOG_FILE"
    ;;
  status)
    if pid="$(running_pid)"; then
      echo "running profile=$PROFILE pid=$pid log=$LOG_FILE"
    else
      echo "stopped profile=$PROFILE log=$LOG_FILE"
    fi
    ;;
  stop)
    pid="$(running_pid)" || { echo "not running profile=$PROFILE"; exit 0; }
    kill -TERM "$pid"
    echo "SIGTERM sent profile=$PROFILE pid=$pid; poll status until the current shard finishes"
    ;;
  kill)
    pid="$(running_pid)" || { echo "not running profile=$PROFILE"; exit 0; }
    kill -KILL "$pid"
    echo "SIGKILL sent profile=$PROFILE pid=$pid"
    ;;
  tail)
    tail -n 80 "$LOG_FILE"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
