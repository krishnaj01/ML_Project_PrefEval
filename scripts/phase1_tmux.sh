#!/usr/bin/env bash
# Launch (or re-attach to) a Phase 1 run inside tmux so it survives terminal/SSH disconnects.
#
#   scripts/phase1_tmux.sh tierA_pilot        # starts configs/phase1/tierA_pilot.yaml in tmux session "prefeval"
#   tmux attach -t prefeval                   # re-attach later;  detach again with  Ctrl-b  then  d
#
# The run resumes automatically if results already exist (same command = resume).
# Environment overrides:  CONDA_ENV=<name> (default: prefeval)   SESSION=<tmux session name> (default: prefeval)
set -euo pipefail
CONFIG_NAME="${1:?usage: $0 <config name, e.g. tierA_pilot>}"
SESSION="${SESSION:-prefeval}"
CONDA_ENV="${CONDA_ENV:-prefeval}"

# Locate conda's shell hook (works for anaconda3 / miniconda3 / miniforge / conda on PATH).
CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
else
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/conda"; do
    [[ -f "$base/etc/profile.d/conda.sh" ]] && { CONDA_SH="$base/etc/profile.d/conda.sh"; break; }
  done
fi
[[ -f "$CONDA_SH" ]] || { echo "Could not find conda (conda.sh). Activate your environment and run: python -m phase1.runner run --config configs/phase1/${CONFIG_NAME}.yaml"; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed (sudo apt install tmux)."; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/phase1/${CONFIG_NAME}.yaml"
[[ -f "$REPO/$CONFIG" ]] || { echo "No such config: $CONFIG"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (a run may be in progress). Attaching..."
  exec tmux attach -t "$SESSION"
fi

# CMD is handed to bash as a single argv element (no extra shell layer), so normal quoting works.
CMD="cd '$REPO' && source '$CONDA_SH' && conda activate '$CONDA_ENV' \
 && python -m phase1.runner run --config $CONFIG"'
 rc=$?; echo; echo "== run ended (exit $rc; 0 = completed, 130 = interrupted, other = error). Summarise with: python -m phase1.aggregate =="; exec bash'
tmux new-session -d -s "$SESSION" -n run bash -lc "$CMD"
tmux new-window -t "$SESSION" -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t "$SESSION:run"
echo "Started '$CONFIG' in tmux session '$SESSION' (windows: run, gpu)."
echo "Attach: tmux attach -t $SESSION   |   Detach: Ctrl-b d   |   Switch window: Ctrl-b n"
