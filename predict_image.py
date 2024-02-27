from ultralytics import YOLO
import yaml
import cv2
import numpy as np
import glob
import pathlib

with open('options.yaml', 'r') as file:
    opts = yaml.safe_load(file)

# Load the object detection model
model = YOLO(opts['model_path'])  # custom YOLOv8n model

# Function to blur pixels inside the obb
def blur_inside_obb(image, obb_vertices_list):
    result = np.copy(image)

    for obb_vertices in obb_vertices_list:
        # Convert the vertices to 32-bit signed integers
        obb_vertices = obb_vertices.astype(np.int32)

        # Create a mask for the OBB region
        obb_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(obb_mask, [obb_vertices], (255, 255, 255))

        # Extract the OBB region
        obb_region = cv2.bitwise_and(image, obb_mask)

        # Apply Gaussian blur to the OBB region
        blurred_obb = cv2.GaussianBlur(obb_region, (51, 51), 0)

        # Invert the mask to get the region outside the OBB
        inverted_mask = cv2.bitwise_not(obb_mask)

        # Extract the region outside the OBB
        outside_obb = cv2.bitwise_and(image, inverted_mask)

        # Combine the blurred OBB region and the region outside the OBB
        result = cv2.add(outside_obb, blurred_obb)

    return result

def blur_in_image(image_in, image_out, save_image = True, show = False):
    obb_vertices_list =  []
    
    # Find the obb vertices
    results = model.predict(image_in, save=False)
    
    image = cv2.imread(image_in)
    for i in range(len(results[0].obb.xyxyxyxy)):
        roi = results[0].obb.xyxyxyxy[i]
        roicpu = roi.detach().cpu()
        obb_vertices = roicpu.numpy().astype(np.int32)
        obb_vertices_list.append(obb_vertices)

    # Apply the blur inside the OBB
    result_image = blur_inside_obb(image, obb_vertices_list)

    # Display the result
    if show == True:
        cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
        cv2.imshow("Blurred", result_image)
        while cv2.getWindowProperty('Blurred', cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(1000)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        pass
    
    if save_image == True:
        cv2.imwrite(image_out, result_image)

        
if __name__== '__main__':
    image_ext = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", "tif", "tiff", ".webp", ".pfm"]
    images = [f for f_ in [glob.glob(opts['input_path']+"*"+e) for e in image_ext] for f in f_]
    for image in images:
       blur_in_image(image_in = image, image_out = opts['output_path'] + pathlib.Path(image).stem +".png",show = opts['show'], save_image=opts['save'])