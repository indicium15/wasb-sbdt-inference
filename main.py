# Example Usage
# python main.py --weights tennis --model deepball --input path/to/input/video.mp4

import argparse
import os
import sys

# Import inference functions from other scripts
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

def main():
    parser = argparse.ArgumentParser(description="Run inference on different models with specified weights and input type.")
    parser.add_argument("--weights", type=str, choices=["tennis", "badminton", "soccer"], required=True, help="Specify the weights to use: 'tennis' or 'badminton' or 'soccer'.")
    parser.add_argument("--model", type=str, choices=list(MODEL_INFERENCE_MAP.keys()), required=True, help="Specify the model to use.")
    parser.add_argument("--input", type=str, required=True, help="Specify the input file or folder.")
    #TODO: add threshold, visualize_output flags

    args = parser.parse_args()

    # Check if input is a file or folder
    if not os.path.exists(args.input):
        print(f"Error: The input path '{args.input}' does not exist.")
        sys.exit(1)

    # Select the appropriate inference function
    inference_function = MODEL_INFERENCE_MAP[args.model]

    # Run the selected inference function with the provided arguments
    inference_function(weights=args.weights, input_path=args.input)

if __name__ == "__main__":
    main()
