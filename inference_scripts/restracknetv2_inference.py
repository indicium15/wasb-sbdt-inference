import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from model_definitions.restracknetv2 import ResTrackNetV2


def preprocess_frame(frame, transform):
    frame = transform(frame)
    return frame


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
        "name": "restracknetv2",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288,
        "inp_width": 512,
        "out_height": 288,
        "out_width": 512,
        "rgb_diff": False,
        "out_scales": [0],
        "halve_channel": False,
        "mode": 'nearest',
        "out_mid_channels": 64,
        "neck_channels": 64,
        "blocks": [3, 3, 4, 3],
        "channels": [16, 32, 64, 128],
        # Update with your model path
        "model_path": f"model_weights/restracknetv2_{weights}_best.pth.tar",
        "video_path": "test_federer.mp4",  # Update with your video path
        # Update with your output video path
        "output_video_path": "test_federer_output_restracknetv2.mp4",
        # Update with your coordinates CSV path
        "coordinates_path": "federer_output_restracknetv2.csv"
    }

    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # Define the transformation to preprocess the frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Load the model and weights
    model = ResTrackNetV2(n_channels=9, n_classes=3, mode=config['mode'],
                        halve_channel=config['halve_channel'],
                        neck_channels=config['neck_channels'],
                        out_mid_channels=config['out_mid_channels'],
                        blocks=config['blocks'],
                        channels=config['channels']).to(device)
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
    output_video_path = os.path.join(os.path.dirname(
        input_path), f"{base_name}_output_restracknetv2.mp4")
    output_csv_path = os.path.join(os.path.dirname(
        input_path), f"{base_name}_output_restracknetv2.csv")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    coordinates = []
    frame_number = 0

    # Buffer to hold the required number of frames for the model
    frames_buffer = []
    prev_positions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) == config['frames_in']:
            # Preprocess the frames
            frames_processed = [preprocess_frame(
                f, transform) for f in frames_buffer]
            input_tensor = torch.cat(
                frames_processed, dim=0).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)[0]  # Get the raw logits

            detected = False
            center_x, center_y, confidence = 0, 0, 0

            for i in range(config['frames_out']):
                output = outputs[0][i]
                # Post-process the output
                # Apply sigmoid to the output to get probabilities
                output = torch.sigmoid(output)
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(
                    heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

                if overlay:
                    heatmap_normalized_visualization = cv2.normalize(
                        heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_normalized_visualization = heatmap_normalized_visualization.astype(
                        np.uint8)
                    # Apply color map to the heatmap
                    heatmap_colored = cv2.applyColorMap(
                        heatmap_normalized_visualization, cv2.COLORMAP_JET)
                    # Overlay the heatmap on the original frame
                    overlayed_frame = cv2.addWeighted(
                        frames_buffer[i], 0.6, heatmap_colored, 0.4, 0)

                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                    (heatmap > 0).astype(np.uint8), connectivity=8)

                # Calculate centers of blobs
                blob_centers = []
                for j in range(1, num_labels):  # Skip the background label 0
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0:
                        center_x = np.sum(
                            np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y = np.sum(
                            np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x, center_y, blob_sum))

                if blob_centers:
                    predicted_position = predict_ball_position(
                        prev_positions, width, height)
                    if predicted_position is not None:
                        # Select the blob closest to the predicted position
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (
                            y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
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
                        cv2.circle(overlayed_frame, (int(center_x), int(
                            center_y)), 10, (0, 255, 0), 2)
                    else:
                        cv2.circle(frames_buffer[i], (int(center_x), int(
                            center_y)), 10, (0, 255, 0), 2)

                # Write the frame to the output video and save the coordinates
                out.write(overlayed_frame if overlay else frames_buffer[0])
                if detected:
                    coordinates.append([frame_number, 1, center_x, center_y, confidence])
                else:
                    coordinates.append([frame_number, 0, 0, 0, 0])

                if overlay:
                    cv2.imshow("Frame", overlayed_frame)
                else:
                    cv2.imshow("Frame", frames_buffer[0])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_number += 1  
            # Clear the buffer for the next set of frames
            frames_buffer = []  

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save coordinates to CSV file
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence (blob size)"])
    coordinates_df.to_csv(output_csv_path, index=False)