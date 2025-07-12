import json
import os

# Create labels directory if it doesn't exist
os.makedirs("labels", exist_ok=True)

# Load JSON data
with open('project-2-at-2025-06-15-13-39-0a361d87.json', 'r') as f:
    data = json.load(f)

# Keypoint label to class index mapping
keypoint_classes = {
    "LT": 0,
    "RT": 1,
    "LB": 2,
    "RB": 3
}

# Process each image annotation
for task in data:
    filename = task["file_upload"]
    base_name = os.path.splitext(filename)[0]  # Remove file extension
    txt_filename = os.path.join("labels", f"{base_name}.txt")
    
    # Get keypoints
    kps = task["annotations"][0]["result"]
    
    # Prepare content for YOLO file
    yolo_lines = []
    
    for kp in kps:
        # Get keypoint label and class index
        label = kp["value"]["keypointlabels"][0]
        class_idx = keypoint_classes[label]
        
        # Get normalized coordinates
        x_center = kp["value"]["x"] / 100.0
        y_center = kp["value"]["y"] / 100.0
        
        # YOLO format: <class> <x_center> <y_center> <width> <height>
        width = height = 0.01  # Fixed small size for keypoints
        yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Write to output file in labels folder
    with open(txt_filename, 'w') as out_file:
        out_file.write("\n".join(yolo_lines))

# print(f"Successfully created {len(data)} label files in 'labels' folder")