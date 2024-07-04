import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.ballseg import BallSeg  

def run_inference(weights, input_path):
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
        "model_path": f"model_weights/ballseg_{weights}_best.pth.tar",  # Update with your model path
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
    model = BallSeg(nclass=1, backbone=config['backbone'], in_channels=6 if config['rgb_diff'] else 3, scale_factors=config['scale_factors']).to(device)
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
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_ballseg.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_ballseg.csv")

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
        
        frame1 = preprocess_frame(frames_buffer[0])
        frame2 = preprocess_frame(frames_buffer[1])

        # Preprocess the frames
        input_tensor = torch.cat([frame1, frame2], dim=0).unsqueeze(0).to(device)

        # Ensure input tensor has the correct shape
        # print(f"Input tensor shape: {input_tensor.shape}")

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)[0]  # Get the raw logits

        # Post-process the output
        output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
        segmentation_map = output.squeeze().cpu().numpy()
        segmentation_map = (segmentation_map > 0.25).astype(np.uint8)  # Apply threshold to get binary map
        segmentation_map = cv2.resize(segmentation_map, (width, height), interpolation=cv2.INTER_NEAREST)
        print(f"Segmentation map shape: {segmentation_map.shape}")
        print(f"Segmentation map unique values: {np.unique(segmentation_map)}")

        # Find coordinates of the tennis ball
        ball_coords = np.column_stack(np.where(segmentation_map == 1))
        if ball_coords.size > 0:
            center_x = np.mean(ball_coords[:, 1])
            center_y = np.mean(ball_coords[:, 0])
            coordinates.append([frame_number, 1, center_x, center_y])

            # Draw a circle on the detected ball
            cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
        else:
            coordinates.append([frame_number,0,0,0])

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
    coordinates_df = pd.DataFrame(coordinates, columns=["frames","detected", "x", "y"])
    coordinates_df.to_csv(output_csv_path, index=False)
