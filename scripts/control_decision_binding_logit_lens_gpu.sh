#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <start|status|stop|kill|tail> [startup|full] [prepared-bundle] [token-audit]"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ACTION="$1"
MODE="${2:-}"
BUNDLE="${3:-}"
TOKEN_AUDIT="${4:-}"
STATE_DIR="${DECISION_LOGIT_LENS_STATE_DIR:-.operator/decision_binding_logit_lens}"
mkdir -p "$STATE_DIR"
PID_FILE="$STATE_DIR/mistral.pid"
MODE_FILE="$STATE_DIR/mistral.mode"
ARGS_FILE="$STATE_DIR/mistral.args"
LOCK_FILE="$STATE_DIR/mistral.lock"

if [[ -z "$MODE" && -f "$MODE_FILE" ]]; then MODE="$(cat "$MODE_FILE")"; fi
MODE="${MODE:-full}"
[[ "$MODE" =~ ^(startup|full)$ ]] || { echo "invalid mode" >&2; exit 2; }
LOG_FILE="$STATE_DIR/mistral-${MODE}.log"

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid command expected_bundle expected_audit owned
  pid="$(cat "$PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    return 1
  fi
  expected_bundle=""
  expected_audit=""
  if [[ -f "$ARGS_FILE" ]]; then
    expected_bundle="$(sed -n '1p' "$ARGS_FILE")"
    expected_audit="$(sed -n '2p' "$ARGS_FILE")"
  fi
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  owned=false
  if [[ "$command" == *"run_decision_binding_logit_lens_gpu.sh"* && \
        "$command" == *"$MODE $expected_bundle $expected_audit"* ]]; then
    owned=true
  elif [[ "$command" == *"decision_binding_logit_lens_cli"* && \
          "$command" == *"--profile mistral"* && \
          "$command" == *"--bundle $expected_bundle"* && \
          "$command" == *"--token-audit $expected_audit"* ]]; then
    owned=true
  fi
  if [[ "$owned" != true ]]; then
    rm -f "$PID_FILE"
    return 1
  fi
  printf '%s' "$pid"
}

case "$ACTION" in
  start)
    [[ -n "$BUNDLE" && -n "$TOKEN_AUDIT" ]] || {
      echo "start requires prepared bundle and token audit" >&2
      exit 2
    }
    if pid="$(running_pid)"; then
      echo "already running pid=$pid" >&2
      exit 3
    fi
    : > "$LOG_FILE"
    echo "$MODE" > "$MODE_FILE"
    printf '%s\n%s\n' "$BUNDLE" "$TOKEN_AUDIT" > "$ARGS_FILE"
    nohup bash -c '
      set -euo pipefail
      exec 9>"$1"
      flock -n 9 || { echo "Mistral logit-lens lock is held" >&2; exit 73; }
      exec scripts/run_decision_binding_logit_lens_gpu.sh "$2" "$3" "$4"
    ' bash "$LOCK_FILE" "$MODE" "$BUNDLE" "$TOKEN_AUDIT" >>"$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"
    echo "started model=mistral mode=$MODE pid=$pid log=$LOG_FILE"
    ;;
  status)
    if pid="$(running_pid)"; then
      echo "running model=mistral mode=$MODE pid=$pid log=$LOG_FILE"
      if [[ -s "$LOG_FILE" ]]; then
        grep '"phase"' "$LOG_FILE" | tail -n 1 || true
      fi
    else
      echo "stopped model=mistral mode=$MODE log=$LOG_FILE"
    fi
    ;;
  stop)
    pid="$(running_pid)" || { echo "not running model=mistral"; exit 0; }
    kill -TERM "$pid"
    echo "SIGTERM sent model=mistral pid=$pid; current atomic chunk will finish"
    ;;
  kill)
    pid="$(running_pid)" || { echo "not running model=mistral"; exit 0; }
    kill -KILL "$pid"
    echo "SIGKILL sent model=mistral pid=$pid"
    ;;
  tail)
    tail -n 100 "$LOG_FILE"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
