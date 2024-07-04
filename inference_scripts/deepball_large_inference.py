import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.deepball import DeepBall

def run_inference(weights, input_path):
    # Configuration parameters
    config = {
        "name": "deepball",
        "frames_in": 1,
        "frames_out": 1,
        "out_scales": [0],
        "class_out": 2,
        "foreground_channel": 1,
        "rgb_diff": False,
        "inp_height": 720,
        "inp_width": 1280,
        "out_height": 180,
        "out_width": 320,
        "block_channels": [32, 64, 128],
        "block_maxpools": [True, True, True],
        "first_conv_kernel_size": 7,
        "first_conv_stride": 2,
        "last_conv_kernel_size": 3,
        "model_path": f"model_weights/deepball-large_{weights}_best.pth.tar",  # Update with your model path
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
    model = DeepBall(n_channels=3, n_classes=config['class_out'], 
                    block_channels=config['block_channels'], 
                    block_maxpools=config['block_maxpools'], 
                    first_conv_kernel_size=config['first_conv_kernel_size'], 
                    first_conv_stride=config['first_conv_stride'], 
                    last_conv_kernel_size=config['last_conv_kernel_size']).to(device)
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
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_deepball_large.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_deepball_large.csv")
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    coordinates = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess_frame(frame).unsqueeze(0).to(device)

        # Ensure input tensor has the correct shape
        print(f"Input tensor shape: {input_tensor.shape}")

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)[0]  # Get the raw logits

        # Post-process the output
        output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
        segmentation_map = output.squeeze().cpu().numpy()
        segmentation_map = (segmentation_map > 0.5).astype(np.uint8)  # Apply threshold to get binary map
        segmentation_map = cv2.resize(segmentation_map, (width, height), interpolation=cv2.INTER_NEAREST)

        # Find coordinates of the tennis ball
        ball_coords = np.column_stack(np.where(segmentation_map == config['foreground_channel']))
        if ball_coords.size > 0:
            center_x = np.mean(ball_coords[:, 1])
            center_y = np.mean(ball_coords[:, 0])
            coordinates.append([frame_number, 1,center_x, center_y])
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
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y"])
    coordinates_df.to_csv(output_csv_path, index=False)