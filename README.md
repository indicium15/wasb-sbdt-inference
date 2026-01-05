# WASB-SBDT Inference Code

## Source Repository
The model weights and code were taken from the [WASB-SBDT GitHub Repository](https://github.com/nttcom/WASB-SBDT/tree/main). Extending their codebase to allow the use of these pre-trained models for inference.  

## Overview
This repository contains scripts for running inference on various models for tracking and segmentation tasks. The main script, main.py, allows users to specify model weights, select a model, and provide an input video file for processing. The models support multiple sports (tennis, badminton, soccer, basketball) and can utilize CUDA, MPS, or CPU for inference.

## Models Supported
1. DeepBall
2. DeepBall-Large
3. BallSeg
4. MonoTrack
5. ResTrackNetV2
6. TrackNetV2
7. WASB

## Download Model Weights
Refer to the [WASB_SBDT Model Zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md) for download links for the model architectures and sports provided above.

## Prerequisites
1. Python 3.x
2. CUDA-enabled GPU (optional but recommended)

## Usage
Install the required packages using `pip install -r requirements.txt`.

The main script, main.py, can be executed from the command line with the following options:

## Sport-Specific Parameter Defaults

The inference system now supports sport-specific parameter defaults through a configuration file (`sport_configs.yaml`). These defaults are automatically applied based on the `--weights` argument, ensuring optimal performance for each sport.

### How It Works

1. **Automatic Defaults**: When you specify `--weights`, the system automatically loads sport-specific parameter defaults from `sport_configs.yaml`
2. **CLI Override**: Command-line arguments always take precedence over config file defaults
3. **Fallback**: If the config file doesn't exist or a sport isn't found, the system falls back to hardcoded defaults

### Default Parameters by Sport

The configuration file (`sport_configs.yaml`) contains optimized defaults for each sport:

- **Badminton**: `gate_px=80.0`, `thr=0.5`, `step=3` (fast, sharp movements, smaller court)
- **Tennis**: `gate_px=110.0`, `thr=0.45`, `step=3` (larger court, similar speed)
- **Soccer**: `gate_px=135.0`, `thr=0.42`, `step=3` (very large field, slower ball)
- **Basketball**: `gate_px=70.0`, `thr=0.52`, `step=3` (fast, close-up, good contrast)

### Customizing Parameters

You can customize parameters in two ways:

1. **Edit the config file**: Modify `sport_configs.yaml` to change defaults for all runs
2. **Use CLI arguments**: Override specific parameters on the command line

**Example**: Use sport defaults but override threshold:
```bash
python main.py --weights tennis --model wasb --input video.mp4 --thr 0.5
```

**Example**: Use custom config file:
```bash
python main.py --weights tennis --model wasb --input video.mp4 --config my_config.yaml
```

### Required Arguments
- `--weights`: Choose between `tennis`, `badminton`, `soccer`, or `basketball`
- `--model`: Select one of the supported models (see list below)
- `--input`: Path to the input video file

### Optional Arguments
- `--overlay`: Display heatmap overlay on output video
- `--show`: Display frames in real-time during inference (press 'q' to quit)
- `--config`: Path to sport configuration file (default: `sport_configs.yaml`)
- `--step`: Step size for MIMO models (1 for oversampling, 3 for non-overlapping windows). Uses sport-specific default if not specified
- `--gate_px`: Gating threshold in pixels for online tracking (MIMO models). Uses sport-specific default if not specified
- `--thr`: Heatmap threshold for blob detection. Uses sport-specific default if not specified
- `--batch_windows`: Batch size for window processing (MIMO models). Uses sport-specific default if not specified

## Run Commands

### WASB (Widely Applicable Strong Baseline)
```bash
# Basic usage
python main.py --weights badminton --model wasb --input path/to/video.mp4

# With real-time visualization
python main.py --weights badminton --model wasb --input path/to/video.mp4 --show

# With heatmap overlay
python main.py --weights badminton --model wasb --input path/to/video.mp4 --overlay

# With oversampling (step=1) for better accuracy
python main.py --weights badminton --model wasb --input path/to/video.mp4 --step 1

# Full example with all options
python main.py --weights badminton --model wasb --input path/to/video.mp4 --overlay --show --step 1 --gate_px 80.0 --thr 0.5 --batch_windows 8
```

### TrackNetV2
```bash
# Basic usage
python main.py --weights badminton --model tracknetv2 --input path/to/video.mp4

# With real-time visualization
python main.py --weights badminton --model tracknetv2 --input path/to/video.mp4 --show

# With oversampling
python main.py --weights badminton --model tracknetv2 --input path/to/video.mp4 --step 1
```

### ResTrackNetV2
```bash
# Basic usage
python main.py --weights badminton --model restracknetv2 --input path/to/video.mp4

# With real-time visualization and overlay
python main.py --weights badminton --model restracknetv2 --input path/to/video.mp4 --show --overlay
```

### MonoTrack
```bash
# Basic usage
python main.py --weights badminton --model monotrack --input path/to/video.mp4

# With custom gating threshold
python main.py --weights badminton --model monotrack --input path/to/video.mp4 --gate_px 100.0
```

### BallSeg
```bash
# Basic usage
python main.py --weights basketball --model ballseg --input path/to/video.mp4

# With heatmap threshold adjustment
python main.py --weights basketball --model ballseg --input path/to/video.mp4 --thr 0.6

# With overlay
python main.py --weights basketball --model ballseg --input path/to/video.mp4 --overlay
```

### DeepBall
```bash
# Basic usage
python main.py --weights soccer --model deepball --input path/to/video.mp4

# With overlay
python main.py --weights soccer --model deepball --input path/to/video.mp4 --overlay
```

### DeepBall-Large
```bash
# Basic usage
python main.py --weights soccer --model deepball-large --input path/to/video.mp4

# With overlay
python main.py --weights soccer --model deepball-large --input path/to/video.mp4 --overlay
```

## Examples by Sport

### Badminton
```bash
# WASB with oversampling
python main.py --weights badminton --model wasb --input test_videos/badminton/test_badminton.mp4 --step 1

# TrackNetV2
python main.py --weights badminton --model tracknetv2 --input test_videos/badminton/test_badminton.mp4
```

### Tennis
```bash
# WASB (uses sport-specific defaults: gate_px=110.0, thr=0.45)
python main.py --weights tennis --model wasb --input path/to/tennis_video.mp4

# WASB with manual override
python main.py --weights tennis --model wasb --input path/to/tennis_video.mp4 --gate_px 120.0

# MonoTrack
python main.py --weights tennis --model monotrack --input path/to/tennis_video.mp4
```

### Soccer
```bash
# DeepBall
python main.py --weights soccer --model deepball --input path/to/soccer_video.mp4

# WASB (uses sport-specific defaults: gate_px=135.0, thr=0.42)
python main.py --weights soccer --model wasb --input path/to/soccer_video.mp4

# DeepBall-Large
python main.py --weights soccer --model deepball-large --input path/to/soccer_video.mp4
```

### Basketball
```bash
# BallSeg
python main.py --weights basketball --model ballseg --input path/to/basketball_video.mp4

# WASB (uses sport-specific defaults: gate_px=70.0, thr=0.52)
python main.py --weights basketball --model wasb --input path/to/basketball_video.mp4
```

## Output Files
For each inference run, the script generates:
- **Output Video**: `{input_name}_output_{model}.mp4` - Video with detected ball positions (green circles)
- **CSV File**: `{input_name}_output_{model}.csv` - Frame-by-frame coordinates with columns: `frame`, `detected`, `x`, `y`, `conf`
## Directory Structure
```
wasb-sbdt-inference/
│
├── main.py
├── sport_configs.yaml          # Sport-specific parameter defaults
├── model_definitions/
│   ├── __init__.py
│   ├── ballseg.py
│   ├── deepball.py
│   ├── deepball_large.py
│   ├── monotrack.py
│   ├── restracknetv2.py
│   ├── tracknetv2.py
│   └── wasb.py
├── inference_scripts/
│   ├── ballseg_inference.py
│   ├── deepball_inference.py
│   ├── deepball_large_inference.py
│   ├── monotrack_inference.py
│   ├── restracknetv2_inference.py
│   ├── tracknetv2_inference.py
│   └── wasb_inference.py
└── model_weights/
    └── badminton_wasb_best.pth.tar
    └── other_model_weights.pth.tar
```

## Model Weights
Place your model weights in the model_weights directory. The script will automatically select the correct weight file based on the specified --weights and --model options.

## GPU Support
The scripts will utilize CUDA for inference if available. If CUDA is not available, the scripts will fall back to using the CPU.

## TODO
- [ ] Add batching to improve inference speeds
- [ ] Implement suggested inference techniques to improve the model performance. Refer to the paper [here](https://arxiv.org/pdf/2311.05237#subsection.3.3) to get a better understanding of this.
