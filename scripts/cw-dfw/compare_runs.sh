#!/bin/bash
# Print the per-role data volumes of a cosmos-rl run and a posttrain run side by side.
#
# The two paths log different things in different places, so "same config" does not imply "same
# data reaches each role". This reads what each run actually did -- episodes generated, ticks
# scored, micro-batches per rank, optimizer steps per rank -- from the logs, which is the only
# evidence that survives a code refactor on either side.
#
# Usage: compare_runs.sh <golden-run-dir> <posttrain-run-dir>
set -euo pipefail

GOLDEN=${1:?golden run dir}
POST=${2:?posttrain run dir}
GL=$GOLDEN/logs/logs_latest

row() { printf "%-34s %-22s %s\n" "$1" "$2" "$3"; }

echo "=== per-role data volumes ==="
row "" "golden (cosmos-rl)" "posttrain"

# Episodes per step. Golden's packer logs once per episode per rank; posttrain's scorer logs the
# whole step's count on the one actor that runs the rollout.
g_eps=$(grep -hc "Packed AlpaGym replay" "$GL"/policy_0.log 2>/dev/null || echo 0)
p_eps=$(cat "$POST"/logs/actor_*.log 2>/dev/null | grep -oE "scored episodes=[0-9]+" | head -1 | grep -oE "[0-9]+" || echo "?")
row "episodes packed (all steps/ranks)" "$g_eps" "${p_eps:-?} per step"

# Ticks per episode -- the simulation length both sides must agree on.
g_tick=$(grep -h "Packed AlpaGym replay" "$GL"/policy_0.log 2>/dev/null | head -1 | grep -oE "valid_steps=[0-9]+" || echo "?")
p_tick=$(cat "$POST"/logs/actor_*.log 2>/dev/null | grep -oE "ticks_per_episode=\[[0-9]+\]" | head -1 || echo "?")
row "ticks per episode" "$g_tick" "$p_tick"

# Micro-batches per rank per step. This is where a missing DP split shows up: posttrain walking
# the whole draw on every rank reads as twice golden's count.
g_batch=$(grep -h "step end" "$GL"/policy_0.log 2>/dev/null | grep rank0 | head -1 | grep -oE "batches=[0-9]+" || echo "?")
p_batch=$(grep -hE "step +[0-9]+/" "$POST"/logs/cosmos_0.log 2>/dev/null | head -1 | grep -oE "train/batches \+[0-9.]+" || echo "(not reported)")
row "micro-batches per rank" "$g_batch" "$p_batch"

echo
echo "=== per-rank divergence (the DP-split witness) ==="
echo "Two ranks drawing the same data report identical numbers here; a real DP split does not."
grep -h "minibatch" "$GL"/policy_0.log 2>/dev/null | head -2 | grep -oE "rank[01]\]|loss=[-0-9.]+" | paste - - | sed 's/^/  golden    /'
for f in "$POST"/logs/actor_*.log; do
  v=$(grep -m1 "first-batch check" "$f" 2>/dev/null | sed 's/^.*check: //')
  [ -n "$v" ] && echo "  posttrain $(basename "$f"): $v"
done
