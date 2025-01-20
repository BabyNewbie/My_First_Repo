import cv2
import torch
import os
import numpy as np
# Pre-trained YOLOv5 Model (small version for speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# path to the folder
image_folder = "C:/Users/mlshe/OneDrive/Desktop/Data_set_train"
output_folder = "C:/Users/mlshe/OneDrive/Desktop/detected_traffic_lights"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def classify_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
    red_mask = red_mask1 | red_mask2
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (36, 50, 50), (85, 255, 255))
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    print(f"Red pixels: {red_pixels}, Yellow pixels: {yellow_pixels}, Green pixels: {green_pixels}")
    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    if red_pixels > max_pixels * 0.5:
        return  "Red"
    elif yellow_pixels > max_pixels * 0.5:
        return "yellow"
    elif green_pixels > max_pixels * 0.5:
        return "green"
    return  "unknown"
# Loop through each image in the folder
for filename in os.listdir(image_folder):
    file_path = os.path.join(image_folder, filename)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(file_path)
        results = model(img)
        results.render()
        img_with_detections = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        labels = results.names
        predictions = results.xyxy[0]
        color_counts = {"Red": 0, "Yellow": 0, "Green": 0}
        # Loop through detections
        for *box, conf, cls in predictions:
            label = labels[int(cls)]
            if label == "traffic light":
                x_min, y_min, x_max, y_max = map(int, box)
                traffic_light_crop = img[y_min:y_max, x_min:x_max]
                if traffic_light_crop.shape[0] < 10 or traffic_light_crop.shape[1] < 10:
                    continue
                traffic_light_crop = cv2.resize(traffic_light_crop, (28, 28))
                traffic_light_crop = cv2.GaussianBlur(traffic_light_crop, (5, 5), 0)
                color = classify_color(traffic_light_crop)
                if color in color_counts:
                    color_counts[color] += 1
                cv2.rectangle(img_with_detections, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_with_detections, color, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        output_path = "C:/Users/mlshe/OneDrive/Desktop/detected_traffic_lights.jpg"
        cv2.imwrite(output_path, img_with_detections)
        print(f"Image with detections saved to {output_path}")
        cv2.imshow("Traffic Light Detection", img_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Traffic Light Color Detected")
        for color, count in color_counts.items():
            print(f"{color}: {count}")