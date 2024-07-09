import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.ballseg import BallSeg  

def preprocess_frame(frame, transform):
    return transform(frame)

def predict_ball_position(prev_positions, width, height):
    if len(prev_positions) < 3:
        return None
    p_t = prev_positions[-1]
    a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
    v_t = p_t - prev_positions[-2] + a_t
    predicted_position = p_t + v_t + 0.5 * a_t
    predicted_position = np.clip(predicted_position, [0, 0], [width, height])
    return predicted_position

def run_inference(weights, input_path, overlay=False):
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
    frames_buffer = []
    prev_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) < config['frames_in']:
            continue

        if len(frames_buffer) > config['frames_in']:
            frames_buffer.pop(0)
        
        frame1 = preprocess_frame(frames_buffer[0], transform)
        frame2 = preprocess_frame(frames_buffer[1], transform)
        frame_diff = preprocess_frame(cv2.absdiff(frames_buffer[1], frames_buffer[0]), transform)

        # Preprocess the frames
        input_tensor = torch.cat([frame2, frame_diff], dim=0).unsqueeze(0).to(device)
        print(input_tensor.shape)

        # Ensure input tensor has the correct shape
        # print(f"Input tensor shape: {input_tensor.shape}")

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)[0]  # Get the raw logits
            print(output.shape)
        
        detected = False
        center_x, center_y, confidence = 0,0,0

        # Post-process the output
        output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
        heatmap = output[0].squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

        if overlay:
            heatmap_normalized_visualization = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_normalized_visualization = heatmap_normalized_visualization.astype(np.uint8)
            # Apply color map to the heatmap
            heatmap_colored = cv2.applyColorMap(heatmap_normalized_visualization, cv2.COLORMAP_JET)
            # Overlay the heatmap on the original frame
            overlayed_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        # Find connected components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)

        # Calculate centers of blobs
        blob_centers = []
        for j in range(1, num_labels):  # Skip the background label 0
            mask = labels_im == j
            blob_sum = heatmap[mask].sum()
            if blob_sum > 0:
                center_x = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                center_y = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                blob_centers.append((center_x, center_y, blob_sum))

        if blob_centers:
            predicted_position = predict_ball_position(prev_positions, width, height)
            if predicted_position is not None:
                # Select the blob closest to the predicted position
                distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                closest_blob_idx = np.argmin(distances)
                center_x, center_y, confidence = blob_centers[closest_blob_idx]
            else:
                # Select the blob with the highest confidence if no prediction is available
                blob_centers.sort(key=lambda x: x[2], reverse=True)
                center_x, center_y, confidence = blob_centers[0]
            detected = True
            prev_positions.append(np.array([center_x, center_y]))
            if len(prev_positions) > 3:
                prev_positions.pop(0)

        # Draw a circle on the detected ball
        if detected:
            if overlay:
                cv2.circle(overlayed_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)

        # Write the frame to the output video and save the coordinates
        out.write(overlayed_frame if overlay else frame)
        if detected:
            coordinates.append([frame_number, 1, center_x, center_y, confidence])
        else:
            coordinates.append([frame_number, 0, 0, 0, 0])

        if overlay:
            cv2.imshow("Frame", overlayed_frame)
        else:
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1  

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save coordinates to CSV file
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence (blob size)"])
    coordinates_df.to_csv(output_csv_path, index=False)