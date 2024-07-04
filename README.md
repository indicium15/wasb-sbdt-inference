# WASB-SBDT Inference Code

## Source Repository
The model weights and code were taken from the [WASB-SBDT GitHub Repository](https://github.com/nttcom/WASB-SBDT/tree/main). Extending their codebase to allow the use of these pre-trained models for inference.  

## Overview
This repository contains scripts for running inference on various models for tracking and segmentation tasks. The main script, main.py, allows users to specify model weights, select a model, and provide an input file or folder for processing. The models are designed to work with either tennis or badminton weights and can utilize CUDA if available.

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

1. weights: Choose between tennis, badminton and football.
2. model: Select one of the supported models.
3. input: Provide the path to the input video file.

## Example Usage
```
python main.py --weights tennis --model deepball --input path/to/input/video.mp4
```
## Directory Structure
```
wasb-sbdt-inference/
│
├── main.py
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
