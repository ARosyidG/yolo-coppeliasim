import os
import time
import csv  # Import CSV module
import cv2
import numpy as np
import urllib.request
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLO

# Load your trained model
model = YOLO('my_model.pt')

# Warping variables
corners = []  # Will store 4 points: [TL, TR, BR, BL]
is_corner_ready = False   
warp_matrix = None
warp_size = (810, 570)  # Output warped image size

# Pixel Position Regression variables
calibration_mode = False
calibration_points = []  # Stores (pixel_x, pixel_y, real_x, real_y)
a1, b1, a2, b2 = None, None, None, None  # Regression parameters

url = "http://192.168.164.192/cam.jpg"
save_dir = "capturedImage"
os.makedirs(save_dir, exist_ok=True)

pause = False

# New: CSV file path for calibration data
CALIBRATION_FILE = "calibration_data.csv"

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global corners, is_corner_ready, warp_matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_corner_ready:
            corners.append((x, y))
            print(f"Added corner {len(corners)} at ({x}, {y})")
            if len(corners) == 4:
                src_pts = np.array(corners, dtype=np.float32)
                dst_pts = np.array([[0, 0], [warp_size[0], 0], 
                                  [warp_size[0], warp_size[1]], [0, warp_size[1]]], dtype=np.float32)
                warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                is_corner_ready = True
                print("Perspective transform matrix calculated. Press 'r' to reset corners.")

def calculate_regression():
    global a1, b1, a2, b2, calibration_points
    if len(calibration_points) < 2:
        print("Need at least 2 calibration points")
        return
    
    # Extract data
    pixel_x = [p[0] for p in calibration_points]
    pixel_y = [p[1] for p in calibration_points]
    real_x = [p[2] for p in calibration_points]
    real_y = [p[3] for p in calibration_points]
    
    # Calculate regression for X-axis
    n = len(pixel_x)
    sum_px = sum(pixel_x)
    sum_rx = sum(real_x)
    sum_px_rx = sum(px * rx for px, rx in zip(pixel_x, real_x))
    sum_px2 = sum(px**2 for px in pixel_x)
    
    b1 = (n * sum_px_rx - sum_px * sum_rx) / (n * sum_px2 - sum_px**2)
    a1 = (sum_rx - b1 * sum_px) / n
    
    # Calculate regression for Y-axis
    sum_py = sum(pixel_y)
    sum_ry = sum(real_y)
    sum_py_ry = sum(py * ry for py, ry in zip(pixel_y, real_y))
    sum_py2 = sum(py**2 for py in pixel_y)
    
    b2 = (n * sum_py_ry - sum_py * sum_ry) / (n * sum_py2 - sum_py**2)
    a2 = (sum_ry - b2 * sum_py) / n
    
    print(f"Regression parameters: X_r = {a1:.2f} + {b1:.4f}*X_p")
    print(f"Regression parameters: Y_r = {a2:.2f} + {b2:.4f}*Y_p")

# New: Save calibration data to CSV
def save_calibration_data():
    global calibration_points, a1, b1, a2, b2
    try:
        with open(CALIBRATION_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header: pixel_x, pixel_y, real_x, real_y
            writer.writerow(['pixel_x', 'pixel_y', 'real_x', 'real_y'])
            for point in calibration_points:
                writer.writerow(point)
            # Write regression parameters
            writer.writerow(['Regression Parameters'])
            writer.writerow(['a1', a1])
            writer.writerow(['b1', b1])
            writer.writerow(['a2', a2])
            writer.writerow(['b2', b2])
        print(f"Calibration data saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration data: {e}")

# New: Load calibration data from CSV
def load_calibration_data():
    global calibration_points, a1, b1, a2, b2
    calibration_points = []
    try:
        with open(CALIBRATION_FILE, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                if row[0] == 'Regression Parameters':
                    # We've reached the parameters section
                    try:
                        # Read the next 4 lines (a1, b1, a2, b2)
                        a1_row = next(reader)
                        b1_row = next(reader)
                        a2_row = next(reader)
                        b2_row = next(reader)
                        
                        a1 = float(a1_row[1])
                        b1 = float(b1_row[1])
                        a2 = float(a2_row[1])
                        b2 = float(b2_row[1])
                    except (StopIteration, IndexError, ValueError) as e:
                        print(f"Error reading parameters: {e}")
                    break
                elif len(row) >= 4:
                    try:
                        pixel_x = float(row[0])
                        pixel_y = float(row[1])
                        real_x = float(row[2])
                        real_y = float(row[3])
                        calibration_points.append((pixel_x, pixel_y, real_x, real_y))
                    except (ValueError, IndexError):
                        print(f"Skipping invalid row: {row}")
        
        print(f"Loaded {len(calibration_points)} calibration points from {CALIBRATION_FILE}")
        if a1 is not None:
            print(f"Regression parameters loaded: X_r = {a1:.6f} + {b1:.6f}*X_p, Y_r = {a2:.6f} + {b2:.6f}*Y_p")
    except FileNotFoundError:
        print(f"Calibration file {CALIBRATION_FILE} not found. Starting fresh.")
    except Exception as e:
        print(f"Error loading calibration data: {e}")

def pixel_to_real(x_pixel, y_pixel):
    if a1 is None or b1 is None or a2 is None or b2 is None:
        return None, None
    x_real = a1 + b1 * x_pixel
    y_real = a2 + b2 * y_pixel
    return x_real, y_real

def main():
    global corners, is_corner_ready, warp_matrix, pause, calibration_mode, calibration_points
    
    # Initialize CoppeliaSim client
    client = RemoteAPIClient()
    sim = client.require('sim')
    
    # Connect to the target object in CoppeliaSim
    target = sim.getObject('/target')
    
    
    # Initialize last frame variables
    last_original_frame = None
    last_warped_frame = None
    last_display_frame = None
    
    cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Camera Feed", mouse_callback)
    
    # New: Load existing calibration data on startup
    load_calibration_data()
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Handle key events
        if key == ord('q'):
            break
        elif key == ord('s'):  # Toggle pause
            pause = not pause
            print(f"Pause is now {pause}")
        elif key == ord('p'):  # Save image
            if last_original_frame is None:
                print("No frame available to save")
            else:
                timestamp = int(time.time() * 1000)
                filename = f"{save_dir}/{timestamp}.jpg"
                if is_corner_ready and last_warped_frame is not None:
                    cv2.imwrite(filename, last_warped_frame)
                else:
                    cv2.imwrite(filename, last_original_frame)
                print(f"Saved image as {filename}")
        elif key == ord('r'):  # Reset corners
            corners = []
            is_corner_ready = False
            warp_matrix = None
            last_warped_frame = None
            print("Corners reset. Please select new points.")
        elif key == ord('c'):  # Toggle calibration mode
            calibration_mode = not calibration_mode
            print(f"Calibration mode {'ON' if calibration_mode else 'OFF'}")
        elif key == ord('a') and calibration_mode:  # Add calibration point
            print(len(current_detections))
            if last_warped_frame is not None:
                # Prompt user for real-world coordinates
                try:
                    real_x = float(input("Enter real-world X coordinate (cm): "))
                    real_y = float(input("Enter real-world Y coordinate (cm): "))
                    # Use last detected object center or mouse position?
                    # For simplicity, we'll use last detected object
                    if len(current_detections) > 0:
                        px, py = current_detections[0]
                        calibration_points.append((px, py, real_x, real_y))
                        print(f"Added calibration point: Pixel({px}, {py}) -> Real({real_x}, {real_y})")
                    else:
                        print("No object detected for calibration")
                except ValueError:
                    print("Invalid input. Numbers only.")
        elif key == ord('f'):  # Finalize calibration
            calculate_regression()
            calibration_mode = False
        # New: Save calibration data with 'w' key
        elif key == ord('w'):
            save_calibration_data()
        
        if pause:
            if last_display_frame is not None:
                cv2.imshow("Camera Feed", last_display_frame)
            continue
            
        try:
            # Fetch frame from IP camera
            img = urllib.request.urlopen(url)
            img_array = np.array(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)
            last_original_frame = frame.copy()
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw existing corners
            for i, pt in enumerate(corners):
                cv2.circle(display_frame, pt, 8, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            # Draw lines between corners
            if len(corners) > 1:
                for i in range(len(corners)-1):
                    cv2.line(display_frame, corners[i], corners[i+1], (0, 255, 255), 2)
                if len(corners) == 4:
                    cv2.line(display_frame, corners[3], corners[0], (0, 255, 255), 2)
            
            # Apply perspective warp if ready
            warped_frame = None
            current_detections = []  # Store centers of detected objects
            
            if is_corner_ready:
                warped_frame = cv2.warpPerspective(frame, warp_matrix, warp_size)
                last_warped_frame = warped_frame.copy()
                results = model(warped_frame)
                display_frame = results[0].plot()
                
                # Extract detection centers
                if results[0].boxes.xyxy is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        current_detections.append((center_x, center_y))
                         
                        # Convert to real-world coordinates
                        x_real, y_real = pixel_to_real(center_x, center_y)
                        # Update target position in CoppeliaSim
                        if sim.getSimulationState() == sim.simulation_advancing_running:
                            if target is not None:
                                targetPosition = sim.getObjectPosition(target, -1)  
                                sim.setObjectPosition(target, -1, [x_real / 40.0, y_real / 40.0, targetPosition[2]])
                            
                        # Display real-world coordinates
                        if x_real is not None and y_real is not None:
                            coord_text = f"({x_real:.1f}, {y_real:.1f})"
                            cv2.putText(display_frame, coord_text, 
                                       (center_x + 20, center_y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                       (0, 255, 0), 2)
            else:
                results = model(frame)
                display_frame = results[0].plot()
                for i, pt in enumerate(corners):
                    cv2.circle(display_frame, pt, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if len(corners) > 1:
                    for i in range(len(corners)-1):
                        cv2.line(display_frame, corners[i], corners[i+1], (0, 255, 255), 2)
                    if len(corners) == 4:
                        cv2.line(display_frame, corners[3], corners[0], (0, 255, 255), 2)
            
            # Show calibration status
            if calibration_mode:
                cv2.putText(display_frame, "CALIBRATION MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Points: {len(calibration_points)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif a1 is not None and b1 is not None:
                cv2.putText(display_frame, "PPR ACTIVE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Store display frame for pause mode
            last_display_frame = display_frame.copy()
            
            # Display frame
            cv2.imshow("Camera Feed", display_frame)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()