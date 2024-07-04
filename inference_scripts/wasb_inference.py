import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.wasb import HRNet  # Replace 'hrnet' with the actual file name

def run_inference(weights, input_path):
    config = {
        "name": "hrnet",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288,
        "inp_width": 512,
        "out_height": 288,
        "out_width": 512,
        "rgb_diff": False,
        "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ['*'],
                "STEM": {
                    "INPLANES": 64,
                    "STRIDES": [1, 1]
                },
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": 'BOTTLENECK',
                    "NUM_BLOCKS": [1],
                    "NUM_CHANNELS": [32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2],
                    "NUM_CHANNELS": [16, 32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE3": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 3,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE4": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 4,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64, 128],
                    "FUSE_METHOD": 'SUM'
                },
                "DECONV": {
                    "NUM_DECONVS": 0,
                    "KERNEL_SIZE": [],
                    "NUM_BASIC_BLOCKS": 2
                }
            },
            "INIT_WEIGHTS": True
        },
        "model_path": f"model_weights/wasb_{weights}_best.pth.tar",  # Update with your model path
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation to preprocess the frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Function to preprocess frames
    def preprocess_frame(frame):
        frame = transform(frame)
        return frame

    # Load the model and weights
    model = HRNet(cfg=config).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()  # Set model to evaluation mode

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.csv")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    coordinates = []
    frame_number = 0

    # Buffer to hold the required number of frames for the model
    frames_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) < config['frames_in']:
            continue

        if len(frames_buffer) > config['frames_in']:
            frames_buffer.pop(0)

        # Preprocess the frames
        frame1 = preprocess_frame(frames_buffer[0])
        frame2 = preprocess_frame(frames_buffer[1])
        frame3 = preprocess_frame(frames_buffer[2])

        input_tensor = torch.cat([frame1, frame2, frame3], dim=0).unsqueeze(0).to(device)

        # Ensure input tensor has the correct shape
        # print(f"Input tensor shape: {input_tensor.shape}")

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)[0]  # Get the raw logits

        for i in range(config['frames_out']):
            output = outputs[0][i]
            # Post-process the output
            output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
            segmentation_map = output.squeeze().cpu().numpy()
            segmentation_map = (segmentation_map > 0.5).astype(np.uint8)  # Apply threshold to get binary map
            segmentation_map = cv2.resize(segmentation_map, (width, height), interpolation=cv2.INTER_NEAREST)

            # Find coordinates of the tennis ball
            ball_coords = np.column_stack(np.where(segmentation_map == 1))
            if ball_coords.size > 0:
                center_x = np.mean(ball_coords[:, 1])
                center_y = np.mean(ball_coords[:, 0])
                coordinates.append([frame_number, 1, center_x, center_y])

                # Draw a circle on the detected ball
                cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
            else:
                coordinates.append([frame_number, 0, 0, 0])

            # Write the frame to the output video
            out.write(frame)
            frame_number += 1

            # Optional: Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save coordinates to CSV file
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number","detected", "x", "y"])
    coordinates_df.to_csv(output_csv_path, index=False)
