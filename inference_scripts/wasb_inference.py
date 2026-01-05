import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pathlib import Path

from model_definitions.wasb import HRNet
from inference_scripts.sbdt_infer_utils import bgr_to_rgb, TrackingConfig, run_mimo_heatmap_inference


def run_inference(weights: str, input_path: str, overlay=False, step=3, gate_px=80.0, thr=0.5, batch_windows=8, show=False):
    cfg_hrnet = {
        "frames_in": 3,
        "frames_out": 3,
        "out_scales": [0],
        "inp_height": 288,
        "inp_width": 512,
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ['*'],
                "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": 'BOTTLENECK',
                           "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": 'SUM'},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": 'BASIC',
                           "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": 'SUM'},
                "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": 'BASIC',
                           "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": 'SUM'},
                "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": 'BASIC',
                           "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": 'SUM'},
                "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
            },
            "INIT_WEIGHTS": True,
        },
        "model_path": f"model_weights/wasb_{weights}_best.pth.tar",
    }

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg_hrnet['inp_height'], cfg_hrnet['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def transform_fn(frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = bgr_to_rgb(frame_bgr)
        return transform(frame_rgb)

    model = HRNet(cfg_hrnet).to(device)
    checkpoint = torch.load(cfg_hrnet['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = Path(input_path).stem
    out_video = os.path.join(os.path.dirname(input_path), f"{base}_output_wasb.mp4")
    out_csv = os.path.join(os.path.dirname(input_path), f"{base}_output_wasb.csv")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    track_cfg = TrackingConfig(thr=thr, gate_px=gate_px, use_gating=True)
    forward_fn = lambda x: model(x)[0]    # expects [B,3,h,w]

    coords = run_mimo_heatmap_inference(
        model=model,
        cap=cap,
        writer=writer,
        transform_fn=transform_fn,
        forward_fn=forward_fn,
        cfg=track_cfg,
        device=device,
        step=step,                 # set to 1 for oversampling
        batch_windows=batch_windows,
        draw=True,
        overlay=overlay,
        show=show,
    )

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(coords, columns=["frame", "detected", "x", "y", "conf"]).to_csv(out_csv, index=False)
