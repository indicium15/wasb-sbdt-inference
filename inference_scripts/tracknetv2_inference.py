import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pathlib import Path
from model_definitions.tracknet2 import TrackNetV2

from inference_scripts.sbdt_infer_utils import bgr_to_rgb, TrackingConfig, run_mimo_heatmap_inference


def run_inference(weights, input_path, overlay=False, step=3, gate_px=80.0, thr=0.5, batch_windows=8, show=False):
    # Configuration parameters
    config = {
        "inp_height": 288,
        "inp_width": 512,
        "bilinear": True,
        "halve_channel": False,
        "mode": "nearest",
        "model_path": f"model_weights/tracknetv2_{weights}_best.pth.tar",
    }

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    # Define the transformation to preprocess the frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def transform_fn(frame_bgr: np.ndarray) -> torch.Tensor:
        # BGR->RGB FIX
        frame_rgb = bgr_to_rgb(frame_bgr)
        return transform(frame_rgb)

    # Load the model and weights
    model = TrackNetV2(
        n_channels=9, n_classes=3,
        bilinear=config['bilinear'],
        mode=config['mode'],
        halve_channel=config['halve_channel']
    ).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = Path(input_path).stem
    out_video = os.path.join(os.path.dirname(input_path), f"{base}_output_tracknetv2.mp4")
    out_csv = os.path.join(os.path.dirname(input_path), f"{base}_output_tracknetv2.csv")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    cfg = TrackingConfig(thr=thr, gate_px=gate_px, use_gating=True)

    # forward_fn must return [B,3,h,w] logits
    forward_fn = lambda x: model(x)[0]

    coords = run_mimo_heatmap_inference(
        model=model,
        cap=cap,
        writer=writer,
        transform_fn=transform_fn,
        forward_fn=forward_fn,
        cfg=cfg,
        device=device,
        step=step,                 # step=1 enables proper oversampling
        batch_windows=batch_windows,
        draw=True,
        overlay=overlay,
        show=show,
    )

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(coords, columns=["frame_number", "detected", "x", "y", "confidence"]).to_csv(out_csv, index=False)
