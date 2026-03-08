#!/usr/bin/env bash
# scripts/eval_libero.sh — Run LIBERO evaluation for a trained policy
set -euo pipefail

POLICY_PATH="${1:?Usage: eval_libero.sh <policy_path> [task_suite] [n_episodes]}"
TASK_SUITE="${2:-libero_object}"
N_EPISODES="${3:-20}"

export MUJOCO_GL=egl

echo "=== Evaluating ${POLICY_PATH} on ${TASK_SUITE} ==="
echo "Episodes per task: ${N_EPISODES}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type=libero \
  --env.task="${TASK_SUITE}" \
  --eval.batch_size=1 \
  --eval.n_episodes="${N_EPISODES}"
