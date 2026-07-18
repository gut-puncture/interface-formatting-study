#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <start|status|stop|kill|tail> <profile> [functional|full] [prepared-bundle]"
  exit 0
fi

ACTION="$1"; PROFILE="$2"; MODE="${3:-}"; BUNDLE="${4:-}"
[[ "$PROFILE" =~ ^(qwen|phi|mistral)$ ]] || { echo "invalid profile" >&2; exit 2; }
STATE_DIR="${DECISION_CONTENT_STATE_DIR:-.operator/decision_binding_content}"
mkdir -p "$STATE_DIR"
PID_FILE="$STATE_DIR/${PROFILE}.pid"; MODE_FILE="$STATE_DIR/${PROFILE}.mode"
ARGS_FILE="$STATE_DIR/${PROFILE}.args"; LOCK_FILE="$STATE_DIR/${PROFILE}.lock"
if [[ -z "$MODE" && -f "$MODE_FILE" ]]; then MODE="$(cat "$MODE_FILE")"; fi
MODE="${MODE:-full}"
[[ "$MODE" =~ ^(functional|full)$ ]] || { echo "invalid mode" >&2; exit 2; }
LOG_FILE="$STATE_DIR/${PROFILE}-${MODE}.log"

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid command expected_bundle
  pid="$(cat "$PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"; return 1
  fi
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  expected_bundle=""; [[ ! -f "$ARGS_FILE" ]] || expected_bundle="$(cat "$ARGS_FILE")"
  if [[ "$command" != *"decision_binding_content"* || "$command" != *"$PROFILE"* || "$command" != *"$expected_bundle"* ]]; then
    rm -f "$PID_FILE"; return 1
  fi
  printf '%s' "$pid"
}

case "$ACTION" in
  start)
    [[ -n "$BUNDLE" ]] || { echo "start requires prepared bundle" >&2; exit 2; }
    if pid="$(running_pid)"; then echo "already running pid=$pid" >&2; exit 3; fi
    : > "$LOG_FILE"; echo "$MODE" > "$MODE_FILE"; echo "$BUNDLE" > "$ARGS_FILE"
    nohup bash -c '
      set -euo pipefail
      exec 9>"$1"; flock -n 9 || exit 73
      exec scripts/run_decision_binding_content_gpu.sh "$2" "$3" "$4"
    ' bash "$LOCK_FILE" "$PROFILE" "$MODE" "$BUNDLE" >>"$LOG_FILE" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "started profile=$PROFILE mode=$MODE pid=$! log=$LOG_FILE"
    ;;
  status)
    if pid="$(running_pid)"; then echo "running profile=$PROFILE pid=$pid log=$LOG_FILE"; else echo "stopped profile=$PROFILE log=$LOG_FILE"; fi
    ;;
  stop)
    pid="$(running_pid)" || { echo "not running profile=$PROFILE"; exit 0; }
    kill -TERM "$pid"; echo "SIGTERM sent; current shard will finish"
    ;;
  kill)
    pid="$(running_pid)" || { echo "not running profile=$PROFILE"; exit 0; }
    kill -KILL "$pid"; echo "SIGKILL sent profile=$PROFILE pid=$pid"
    ;;
  tail) tail -n 100 "$LOG_FILE" ;;
  *) echo "invalid action" >&2; exit 2 ;;
esac
