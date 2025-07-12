import cv2
import numpy as np
import os
import random
import math
from pathlib import Path

# Configuration
DATASET_DIR = "DatasetCorner"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
TARGET_COUNT = 300
MIN_ANGLE = 5
MAX_ANGLE = 15

# Create directories if needed
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

# Get current image files
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
current_count = len(image_files)
print(f"Current images: {current_count}, Target: {TARGET_COUNT}")

# Calculate how many rotations we need per image
rotations_per_image = max(1, (TARGET_COUNT - current_count) // current_count + 1)
print(f"Generating {rotations_per_image} rotations per image")

# Keypoint class mapping
keypoint_classes = {"LT": 0, "RT": 1, "LB": 2, "RB": 3}

def rotate_point(x, y, angle, cx, cy):
    """Rotate a point around a center point"""
    radians = math.radians(angle)
    cos = math.cos(radians)
    sin = math.sin(radians)
    
    # Translate point to origin
    x -= cx
    y -= cy
    
    # Rotate point
    nx = x * cos - y * sin
    ny = x * sin + y * cos
    
    # Translate back
    return nx + cx, ny + cy

def rotate_image_and_keypoints(image_path, label_path, angle):
    """Rotate image and its keypoints by specified angle"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    # Get image dimensions
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(0, 0, 0))
    
    # Load keypoints
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    rotated_keypoints = []
    for line in lines:
        parts = line.strip().split()
        class_idx = int(parts[0])
        x = float(parts[1]) * w
        y = float(parts[2]) * h
        
        # Rotate keypoint
        nx, ny = rotate_point(x, y, angle, cx, cy)
        
        # Convert back to normalized coordinates
        nx_norm = nx / w
        ny_norm = ny / h
        
        # Ensure within bounds
        nx_norm = max(0.0, min(1.0, nx_norm))
        ny_norm = max(0.0, min(1.0, ny_norm))
        
        rotated_keypoints.append([class_idx, nx_norm, ny_norm, 0.01, 0.01])
    
    return rotated_image, rotated_keypoints

# Augmentation process
for img_file in image_files[:]:  # Use copy since we'll be adding files
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, base_name + ".txt")
    
    # Skip if label doesn't exist
    if not os.path.exists(label_path):
        print(f"Skipping {img_file} - label not found")
        continue
    
    for i in range(rotations_per_image):
        # Generate random angle (positive or negative)
        angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
        angle *= random.choice([-1, 1])  # Random direction
        
        # Rotate image and keypoints
        rotated_img, rotated_kps = rotate_image_and_keypoints(img_path, label_path, angle)
        
        if rotated_img is None:
            continue
            
        # Create new filenames
        new_base = f"{base_name}_rot{i}_{int(angle)}"
        new_img_file = new_base + os.path.splitext(img_file)[1]
        new_label_file = new_base + ".txt"
        
        # Save rotated image
        new_img_path = os.path.join(IMAGES_DIR, new_img_file)
        cv2.imwrite(new_img_path, rotated_img)
        
        # Save rotated keypoints
        new_label_path = os.path.join(LABELS_DIR, new_label_file)
        with open(new_label_path, 'w') as f:
            for kp in rotated_kps:
                f.write(f"{kp[0]} {kp[1]:.6f} {kp[2]:.6f} {kp[3]:.6f} {kp[4]:.6f}\n")
        
        print(f"Created: {new_img_file} (rotated {angle:.1f}°)")

print("Augmentation complete!")
print(f"Final image count: {len(os.listdir(IMAGES_DIR))}")
print(f"Final label count: {len(os.listdir(LABELS_DIR))}")