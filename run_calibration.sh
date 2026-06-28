#!/usr/bin/env bash
set -e

echo "=================================================="
echo "Running overnight speaker calibration limits"
echo "=================================================="

# Ensure the xtts environment is activated, since we need PyTorch + SpeechBrain
VENV_DIR="overnight_runs/.venv_xtts"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: $VENV_DIR not found. Run ./run_fleurs_experiment_example.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# We only need to compute calibration limits ONCE because the "different speaker"
# and "generated vs wrong reference" bounds are largely a property of the model's output
# distribution and the reference distribution. But we can compute it for each model.

for run_dir in overnight_runs/results_*; do
    if [ -d "$run_dir" ] && [ -f "$run_dir/manifest.json" ]; then
        echo ""
        echo "Calibrating run: $run_dir"
        python3 xttslab.py calibrate --run "$run_dir"
    fi
done

echo ""
echo "All calibrations complete!"
