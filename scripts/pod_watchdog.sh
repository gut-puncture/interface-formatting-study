#!/bin/zsh
# Mac-side watchdog for one Prime Intellect pod.
# Every 5 minutes: copy results back. When the run finishes (ALL.done) or the hard time limit
# passes, copy everything back, verify the copy, and terminate the pod.
# usage: pod_watchdog.sh POD_ID SSH_TARGET SSH_PORT KEY REMOTE_RUN_DIR LOCAL_RUN_DIR HARD_LIMIT_EPOCH
POD=$1; TARGET=$2; PORT=$3; KEY=$4; RDIR=$5; LDIR=$6; LIMIT=$7
KH=$HOME/.ssh/known_hosts_keyhole
SSH="ssh -i $KEY -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -o ConnectTimeout=20"
mkdir -p "$LDIR"
LOG="$LDIR/watchdog.log"
sync_back() {
  rsync -az --partial -e "$SSH" "$TARGET:$RDIR/" "$LDIR/" >> "$LOG" 2>&1
}
verify() {
  # every remote file must exist locally with the same size
  ${=SSH} "$TARGET" "cd $RDIR && find . -type f -printf '%s %p\n' | sort -k2" > "$LDIR/.remote_list" 2>>"$LOG" || return 1
  (cd "$LDIR" && find . -type f ! -name '.remote_list' ! -name 'watchdog.log' -exec stat -f '%z %N' {} \; | sort -k2) > "$LDIR/.local_list"
  python3 - "$LDIR" <<'EOF'
import sys
d = sys.argv[1]
r = dict(l.split(" ", 1)[::-1] for l in open(d + "/.remote_list").read().splitlines() if l.strip())
l = dict(x.split(" ", 1)[::-1] for x in open(d + "/.local_list").read().splitlines() if x.strip())
bad = [k for k, v in r.items() if k not in ("./watchdog.log",) and l.get(k) != v]
print("missing_or_different", len(bad), bad[:5])
sys.exit(1 if bad else 0)
EOF
}
terminate() {
  echo "$(date) terminating $POD" >> "$LOG"
  prime --plain pods terminate "$POD" --yes >> "$LOG" 2>&1
  sleep 30
  prime --plain pods list >> "$LOG" 2>&1
}
while true; do
  sync_back
  NOW=$(date +%s)
  if ${=SSH} "$TARGET" "test -f $RDIR/ALL.done" 2>/dev/null || [ "$NOW" -gt "$LIMIT" ]; then
    echo "$(date) finishing (done or limit)" >> "$LOG"
    for i in 1 2 3; do sync_back; verify >> "$LOG" 2>&1 && break; sleep 20; done
    if verify >> "$LOG" 2>&1 || [ "$NOW" -gt $((LIMIT + 1800)) ]; then
      terminate; echo "$(date) watchdog exit" >> "$LOG"; exit 0
    fi
  fi
  sleep 300
done
