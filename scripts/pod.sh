#!/bin/bash
# Helpers for the rented pod. usage: pod.sh ssh "<cmd>" | pod.sh push | pod.sh pull
HOST=${POD_HOST:-ubuntu@216.81.200.46}
KEY=$HOME/.ssh/twc_c3a_20260720_ed25519
OPTS="-i $KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$HOME/.ssh/known_hosts_keyhole -o ConnectTimeout=20 -o ServerAliveInterval=30"
REPO=$(cd "$(dirname "$0")/.." && pwd)
case "$1" in
  ssh)  shift; ssh $OPTS "$HOST" "$@" ;;
  push) rsync -az -e "ssh $OPTS" "$REPO/keyhole" "$REPO/data" "$REPO/scripts" "$REPO/tests" "$HOST:~/kh/" ;;
  pull) mkdir -p "$REPO/runs/main"; rsync -az --partial -e "ssh $OPTS" "$HOST:~/kh/runs/main/" "$REPO/runs/main/" ;;
esac
