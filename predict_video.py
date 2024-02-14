from ultralytics import YOLO
import torch
import cv2
import numpy as np
import yaml

with open('paths.yaml', 'r') as file:
    paths = yaml.safe_load(file)

# Load a model
model = YOLO(paths['model_path'])  # custom YOLOv8n obb model


def blur_inside_obbs(frame, obb_vertices_list):
    result_frame = np.zeros_like(frame, dtype=np.uint8)

    for obb_vertices in obb_vertices_list:
        # Convert the vertices to 32-bit signed integers
        obb_vertices = obb_vertices.astype(np.int32)

        # Create a mask for the OBB region
        obb_mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(obb_mask, obb_vertices_list, (255, 255, 255))

        # Extract the OBB region
        obb_region = cv2.bitwise_and(frame, obb_mask)

        # Apply Gaussian blur to the OBB region
        blurred_obb = cv2.blur(obb_region, (35, 35))

        # Invert the mask to get the region outside the OBB
        inverted_mask = cv2.bitwise_not(obb_mask)

        # Extract the region outside the OBB
        outside_obb = cv2.bitwise_and(frame, inverted_mask)

        # Combine the blurred OBB region and the region outside the OBB
        result_frame = cv2.add(outside_obb, blurred_obb)

    return result_frame


# Open a video capture object
cap = cv2.VideoCapture(paths['input_video_path'])

# Predict the obb for each frame
results = model.predict(paths['input_video_path'], save=False, conf = 0.2)

obb_vertices_list = []

#Creating a list of obb vertices for a single frame. The list is appended to the obb_ver_list and then emptied to receive the next frame.
obb_vertices_frame = []

# Creating a list of lists. Each sublist contains the obb vertices of a single frame
obb_ver_list = []

#Each obb vertices included to a frame are saved as a list to the obb_ver_list
for i in range(len(results)):
    obb_ver_list.append(obb_vertices_frame)
    obb_vertices_frame = []   
    for j in range(len(results[i].obb.xyxyxyxy)):
        roi = results[i].obb.xyxyxyxy[j]
        roi_int = roi.type(torch.int32)
        roicpu = roi_int.detach().cpu()
        obb_vertices = roicpu.numpy()
        obb_vertices_frame.append(obb_vertices)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(paths['output_video_path'], fourcc, fps, (width, height))

frame_index = 1

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    # Get OBB vertices for the current frame
    obb_vertices_list = obb_ver_list[frame_index]

    # Apply the blur inside each OBB to the current frame
    if obb_vertices_list != []:
        result_frame = blur_inside_obbs(frame, obb_vertices_list)
    else:
        result_frame = frame

    # Write the frame to the output video file
    out.write(result_frame)

    # Display the result frame
    cv2.imshow("Blurred Video", result_frame)

    # Increment the frame index
    frame_index += 1

    # Break the loop if 'q' key is pressed or if all frames are processed
    if cv2.waitKey(30) & 0xFF == ord('q') or frame_index >= len(obb_ver_list):
        break

# Release the video capture and writer objects, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()