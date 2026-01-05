import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.ballseg import BallSeg

from inference_scripts.sbdt_infer_utils import bgr_to_rgb, extract_blob_candidates_coh, select_with_gating_maxconf


def run_inference(weights, input_path, overlay=False, thr=0.5):
    # Configuration parameters
    config = {
        "name": "ballseg",
        "frames_in": 2,
        "frames_out": 1,
        "out_scales": [0],
        "rgb_diff": True,
        "inp_height": 576,
        "inp_width": 1024,
        "out_height": 576,
        "out_width": 1024,
        "backbone": "resnet18",
        "scale_factors": [1, 1, 0.5],
        "model_path": f"model_weights/ballseg_{weights}_best.pth.tar",
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation to preprocess the frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the model and weights
    model = BallSeg(
        nclass=1,
        backbone=config['backbone'],
        in_channels=6 if config['rgb_diff'] else 3,
        scale_factors=config['scale_factors']
    ).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_ballseg.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_ballseg.csv")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    coordinates = []
    frame_number = 0
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) < config['frames_in']:
            continue

        if len(frames_buffer) > config['frames_in']:
            frames_buffer.pop(0)

        # BGR->RGB conversion for preprocessing
        frame1_rgb = bgr_to_rgb(frames_buffer[0])
        frame2_rgb = bgr_to_rgb(frames_buffer[1])
        frame_diff = cv2.absdiff(frames_buffer[1], frames_buffer[0])
        frame_diff_rgb = bgr_to_rgb(frame_diff)

        frame1 = transform(frame1_rgb)
        frame2 = transform(frame2_rgb)
        frame_diff_t = transform(frame_diff_rgb)

        # Preprocess the frames
        input_tensor = torch.cat([frame2, frame_diff_t], dim=0).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)[0]  # Get the raw logits

        # Post-process the output
        prob = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)  # [H,W]
        prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)

        # Use blob-based CoH extraction
        cands = extract_blob_candidates_coh(prob, thr=thr)
        det, cx, cy, conf = select_with_gating_maxconf(cands, pred_xy=None, gate_px=None)

        # Write the frame corresponding to the detection (frames_buffer[1])
        frame_out = frames_buffer[1].copy()
        
        # Apply overlay if enabled
        if overlay:
            heatmap_normalized = cv2.normalize(prob, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_normalized = heatmap_normalized.astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            frame_out = cv2.addWeighted(frame_out, 0.6, heatmap_colored, 0.4, 0)
        
        if det:
            coordinates.append([frame_number, 1, cx, cy, conf])
            cv2.circle(frame_out, (int(cx), int(cy)), 10, (0, 255, 0), 2)
        else:
            coordinates.append([frame_number, 0, 0, 0, 0])

        # Write the frame to the output video
        out.write(frame_out)
        frame_number += 1

        # Optional: Display the frame
        cv2.imshow("Frame", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save coordinates to CSV file
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence"])
    coordinates_df.to_csv(output_csv_path, index=False)
