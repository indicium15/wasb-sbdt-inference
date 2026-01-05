import argparse
import os
import sys
from pathlib import Path
import yaml
from inference_scripts.deepball_inference import run_inference as deepball_inference
from inference_scripts.deepball_large_inference import run_inference as deepball_large_inference
from inference_scripts.ballseg_inference import run_inference as ballseg_inference
from inference_scripts.monotrack_inference import run_inference as monotrack_inference
from inference_scripts.restracknetv2_inference import run_inference as restracknetv2_inference
from inference_scripts.tracknetv2_inference import run_inference as tracknetv2_inference
from inference_scripts.wasb_inference import run_inference as wasb_inference

# Map models to their corresponding inference functions
MODEL_INFERENCE_MAP = {
    "deepball": deepball_inference,
    "deepball-large": deepball_large_inference,
    "ballseg": ballseg_inference,
    "monotrack": monotrack_inference,
    "restracknetv2": restracknetv2_inference,
    "tracknetv2": tracknetv2_inference,
    "wasb": wasb_inference,
}

def load_sport_config(config_path: str, sport: str) -> dict:
    """
    Load sport-specific configuration from YAML file.
    Returns empty dict if file doesn't exist or sport not found.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if config and sport in config:
                return config[sport]
    except (yaml.YAMLError, KeyError) as e:
        print(f"Warning: Could not load config for sport '{sport}': {e}")
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run inference on different models with specified weights and input type.")
    parser.add_argument("--weights", type=str, choices=["tennis", "badminton", "soccer", "basketball"], required=True)
    parser.add_argument("--model", type=str, choices=list(MODEL_INFERENCE_MAP.keys()), required=True)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--show', action='store_true', help="Display frames in real-time during inference (press 'q' to quit)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, default="sport_configs.yaml", help="Path to sport configuration file (default: sport_configs.yaml)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference (legacy, maps to batch_windows for MIMO models)")
    parser.add_argument("--step", type=int, default=None, choices=[1, 3], help="Step size: 1 for oversampling, 3 for non-overlapping windows")
    parser.add_argument("--gate_px", type=float, default=None, help="Gating threshold in pixels for online tracking")
    parser.add_argument("--thr", type=float, default=None, help="Heatmap threshold for blob detection")
    parser.add_argument("--batch_windows", type=int, default=None, help="Batch size for window processing (MIMO models)")

    args = parser.parse_args()
    
    # Load sport-specific defaults from config file
    sport_config = load_sport_config(args.config, args.weights)
    
    # Apply defaults: use config if CLI arg is None, otherwise use CLI arg
    # Fallback to hardcoded defaults if neither config nor CLI provides value
    step = args.step if args.step is not None else sport_config.get('step', 3)
    gate_px = args.gate_px if args.gate_px is not None else sport_config.get('gate_px', 80.0)
    thr = args.thr if args.thr is not None else sport_config.get('thr', 0.5)
    batch_windows = args.batch_windows if args.batch_windows is not None else sport_config.get('batch_windows', 8)

    if not os.path.exists(args.input):
        print(f"Error: The input path '{args.input}' does not exist.")
        sys.exit(1)

    inference_function = MODEL_INFERENCE_MAP[args.model]

    # Handle legacy batch_size parameter
    if args.batch_size is not None:
        batch_windows = args.batch_size

    # Determine which models are MIMO (need step, gate_px, etc.)
    mimo_models = ["tracknetv2", "restracknetv2", "monotrack", "wasb"]
    
    if args.model in mimo_models:
        inference_function(
            weights=args.weights,
            input_path=args.input,
            overlay=args.overlay,
            step=step,
            gate_px=gate_px,
            thr=thr,
            batch_windows=batch_windows,
            show=args.show
        )
    elif args.model == "ballseg":
        inference_function(
            weights=args.weights,
            input_path=args.input,
            overlay=args.overlay,
            thr=thr
        )
    else:
        # For deepball models, pass thr and show parameters
        inference_function(
            weights=args.weights,
            input_path=args.input,
            overlay=args.overlay,
            thr=thr,
            show=args.show
        )

if __name__ == "__main__":
    main()
