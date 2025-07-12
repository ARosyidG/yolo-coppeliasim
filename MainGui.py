import math
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
import threading
import queue
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import urllib.request

# Initialize global variables
corners = []  # Stores 4 points: [TL, TR, BR, BL]
is_corner_ready = False   
calibration_mode = False
calibration_points = []  # Stores (pixel_x, pixel_y, real_x, real_y)
a1, b1, a2, b2 = None, None, None, None  # Regression parameters
current_object_pos = None  # Stores current detected object position
warp_size = (810, 570)  # Output warped image size
CALIBRATION_FILE = "calibration_data.csv"
warp_matrix = None
pause = False

# Initialize CoppeliaSim client
client = RemoteAPIClient()
sim = client.require('sim')
simIK = client.require('simIK')
target = sim.getObject('/target')

vidURL = "http://192.168.164.192/cam.jpg"

# Create a thread-safe queue for frame sharing
frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory bloat

# Initialize YOLO model
try:
    model = YOLO('my_model.pt')
except:
    model = None
    print("YOLO model not found. Detection disabled.")
    
# DH Parameters
dh_params = [
    {'d': 0.0661, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': math.pi/2},
    {'d': 0.0000, 'a': 0.4251, 'alpha': 0, 'offset': 0},
    {'d': 0.0000, 'a': 0.3922, 'alpha': 0, 'offset': 0},
    {'d': 0.0397, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': -math.pi/2},
    {'d': 0.0492, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': math.pi/2},
    {'d': 0.0000, 'a': 0.0000, 'alpha': 0, 'offset': 0}
]

# Inverse kinematics function
def inverse_kinematics(target_position):
    """
    Calculate joint angles to reach target position
    target_position: [x, y, z] in meters
    Returns: list of joint angles in radians or None if unreachable
    """
    # Extract DH parameters
    d1 = dh_params[0]['d']
    a1_dh = dh_params[1]['a']
    a2_dh = dh_params[2]['a']
    d5 = dh_params[4]['d']
    
    # Assume end effector pointing down (approach vector = [0, 0, -1])
    wrist_center = [
        target_position[0],
        target_position[1],
        target_position[2] + d5  # Wrist center is above target by d5
    ]
    
    # Joint 1 (theta1)
    wx, wy, wz = wrist_center
    theta1 = math.atan2(wy, wx) - dh_params[0]['offset']
    
    # Projection on XY plane
    r = math.sqrt(wx**2 + wy**2)
    z_planar = wz - d1
    
    # Distance from joint2 to wrist center
    D = math.sqrt(r**2 + z_planar**2)
    
    # Check if position is reachable
    if D > a1_dh + a2_dh or D < abs(a1_dh - a2_dh):
        return None
    
    # Joint 3 (theta3) - elbow up solution
    cos_theta3 = (a1_dh**2 + a2_dh**2 - D**2) / (2 * a1_dh * a2_dh)
    if abs(cos_theta3) > 1:
        return None
    theta3 = math.acos(cos_theta3) - math.pi  # Elbow up configuration
    raw_theta3 = theta3 - dh_params[2]['offset']
    
    # Joint 2 (theta2)
    alpha = math.atan2(z_planar, r)
    beta = math.atan2(a2_dh * math.sin(theta3), a1_dh + a2_dh * math.cos(theta3))
    theta2 = alpha - beta - dh_params[1]['offset']
    
    # For simplicity, assume fixed orientation (pointing down)
    # Joint 4, 5, 6
    theta4 = 0 - dh_params[3]['offset']  # -offset
    theta5 = math.pi/2 - dh_params[4]['offset']  # Pointing down
    theta6 = 0 - dh_params[5]['offset']
    
    return [theta1, theta2, raw_theta3, theta4, theta5, theta6]

# Video capture thread function
def video_capture_thread(url):
    while True:
        try:
            img = urllib.request.urlopen(url)
            img_array = np.array(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)
            
            # Resize and convert to RGB
            frame = cv2.resize(frame, warp_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Put frame in queue, drop old frames if queue is full
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass
            try:
                frame_queue.put(frame, timeout=0.5)
            except queue.Full:
                pass
        except Exception as e:
            print(f"Camera error: {e}")
            time.sleep(1)  # Wait before retrying

# Mouse callback function
def mouse_callback(event):
    global corners, is_corner_ready
    if not is_corner_ready and len(corners) < 4:
        x, y = event.x, event.y
        corners.append((x, y))
        app.status_var.set(f"Added corner {len(corners)} at ({x}, {y})")
        
        if len(corners) == 4:
            src_pts = np.array(corners, dtype=np.float32)
            dst_pts = np.array([[0, 0], [warp_size[0], 0], 
                              [warp_size[0], warp_size[1]], [0, warp_size[1]]], dtype=np.float32)
            global warp_matrix
            warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            is_corner_ready = True
            app.status_var.set("Perspective transform matrix calculated. Press 'r' to reset corners.")

# Create main application window
class ImageDetectionAPP:
    def __init__(self, root):
        global vidURL
        
        self.DebuggingValue = {"x" : 0,"y" : 0}
        self.isDebugging = tk.BooleanVar(value=False)
        
        # option to use SimIK Coppeliasim or Manual Calculation
        self.Modes = ["SimIK", "Manual"]        
        self.current_Mode = tk.StringVar(value=self.Modes[0])
        
               
        # SimIK Set-UP
        UR5Base = sim.getObject('/UR5')
        UR5Tip = sim.getObject('/connection')
        self.IKEnvorement = simIK.createEnvironment()
        self.IKGroup = simIK.createGroup(self.IKEnvorement)
        ikElement,simToIkObjectMap,ikToSimObjectMap=simIK.addElementFromScene(self.IKEnvorement,self.IKGroup,UR5Base,UR5Tip,target,simIK.constraint_position)
        
        
        self.vidURL = vidURL
        self.joint_handles = []
        self.root = root
        self.root.title("ARM Controller And Image Detection")
        self.root.geometry("1200x650")
        self.root.minsize(1200, 650)
        self.current_frame = None  # Store the latest raw frame
        self.warped_frame = None  # Store the latest warped frame
        self.current_detections = []  # Store centers of detected objects
        
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        
        self.upper_frame = tk.Frame(root, height=570, bd=2, relief="groove")
        self.upper_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.upper_frame.grid_columnconfigure(1, weight=1)
        
        # Create image display area
        self.image_frame = tk.Frame(self.upper_frame, width=810, height=570, bd=2, relief="groove")
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.image_frame.grid_propagate(False)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Button-1>", mouse_callback)
        
        # Placeholder image
        self.placeholder = Image.new("RGB", (810, 570), (50, 100, 150))
        self.placeholder_img = ImageTk.PhotoImage(self.placeholder)
        self.image_label.config(image=self.placeholder_img)
        
        # Create control panel
        self.control_frame = tk.LabelFrame(root, text="Controls")
        self.control_frame.grid(row=1, column=0, padx=10, sticky="nsew")
        
        # Create slider area
        self.create_slider_area()
        # Create widgets
        self.create_widgets()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        # Load existing calibration data if available
        self.load_calibration_data()
        
        # Start video capture thread
        self.capture_thread = threading.Thread(
            target=video_capture_thread, 
            args=(self.vidURL,),
            daemon=True
        )
        self.capture_thread.start()
        
        # Start frame processing
        self.update_video_feed()

    def create_slider_area(self):
        # Create frame for slider controls
        self.slider_frame = tk.LabelFrame(self.upper_frame, text="Joint Controls", padx=10, pady=10)
        self.slider_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.slider_frame.grid_columnconfigure(0, weight=1)
        
        # Create sliders and labels
        self.jointslidersVar = []
        self.sliders = []
        for i in range(6):
            joint_path = "/joint{"+str(i)+"}"  # Adjust path to match your scene hierarchy
            try:
                handle = sim.getObject(joint_path)
                self.joint_handles.append(handle)
            except Exception as e:
                print(f"error: {e}")
                
        joint_names = [
            "Joint 1", "Joint 2", "Joint 3", 
            "Joint 4", "Joint 5", "Joint 6"
        ]
        
        for i, name in enumerate(joint_names):
            # Create frame for each slider
            slider_container = tk.Frame(self.slider_frame, relief="groove")
            slider_container.grid(row=i, column=0, sticky="ew", pady=5)
            
            # Create label
            tk.Label(slider_container, text=name).pack(side="left")
            
            # Create slider
            self.jointslidersVar.append(tk.DoubleVar(value=0)) 
            slider = tk.Scale(
                slider_container, 
                from_=-180, 
                to=180, 
                orient="horizontal",
                variable=self.jointslidersVar[i],
                length=200,
                showvalue=True,
                resolution=0.1
            )
            slider.pack(side="left", fill="x", expand=True)
            slider.configure(command=lambda value, idx=i: self.slider_moved(idx, float(value)))
            self.sliders.append(slider)

        
        modeDropDown = ttk.Combobox(self.slider_frame, values=self.Modes, state="readonly", textvariable=self.current_Mode)
        modeDropDown.bind("<<ComboboxSelected>>", self.changeMode)
        modeDropDown.grid(row=6, column=0, sticky="ew")
        
        self.KinematicFrame = tk.Frame(self.slider_frame, relief="groove")
        self.KinematicFrame.grid(row=7, column=0, sticky="ew")
        self.KinematicFrame.grid_columnconfigure(0, weight=1)
        self.KinematicFrame.grid_columnconfigure(1, weight=0)
        
        
        # debuging sliders
        self.DebugingSlidersFrame = tk.LabelFrame(self.KinematicFrame, text="Debugging slider", padx=10, pady=10)
        self.DebugingSlidersFrame.grid_columnconfigure(0, weight=1)
        isDebuggingCheckbox = tk.Checkbutton(self.DebugingSlidersFrame, text="is debugging", variable=self.isDebugging, command=self.showIsDebugingValue)
        isDebuggingCheckbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        debugingSliderContainerX = tk.Frame(self.DebugingSlidersFrame, relief="groove")
        debugingSliderContainerX.grid(row=1, column=0, sticky="nsew", pady=5)
        tk.Label(debugingSliderContainerX, text="X").pack(side="left")
        varX = tk.DoubleVar(value=0)
        sliderX = tk.Scale(
            debugingSliderContainerX, 
            from_=-1, 
            to=1, 
            orient="horizontal",
            variable=varX,
            length=200,
            showvalue=True,
            resolution=0.01
        )
        sliderX.pack(side="left", fill="x", expand=True)
        debugingSliderContainerY = tk.Frame(self.DebugingSlidersFrame, relief="groove")
        debugingSliderContainerY.grid(row=2, column=0, sticky="nsew", pady=5)
        tk.Label(debugingSliderContainerY, text="Y").pack(side="left")
        varY = tk.DoubleVar(value=0)
        sliderY = tk.Scale(
            debugingSliderContainerY, 
            from_=-1, 
            to=1, 
            orient="horizontal",
            variable=varY,
            length=200,
            showvalue=True,
            resolution=0.01
        )
        sliderY.pack(side="left", fill="x", expand=True)
        sliderX.configure(command=lambda value: self.setDebuggingSliders("x", float(value)))
        sliderY.configure(command=lambda value: self.setDebuggingSliders("y", float(value)))
        
        
        # Create "Go to Target" button
        self.goto_button = tk.Button(
            self.KinematicFrame, 
            text="Go to Target",
            command=self.goto_target_callback,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.goto_button.grid(row=0, column=0, sticky="nsew")
    
    
    def changeMode(self, event):
        currentMode = self.current_Mode.get()
        if currentMode == "SimIK":
            self.DebugingSlidersFrame.grid_remove()
            self.goto_button.grid(row=0, column=0, sticky="nsew", pady=15)
            self.KinematicFrame.grid_columnconfigure(1, weight=0)
        elif currentMode == "Manual":
            self.DebugingSlidersFrame.grid(row=0, column=0, sticky="ew", pady=5)
            self.goto_button.grid(row=0, column=1, sticky="nsew", pady=15)
            self.KinematicFrame.grid_columnconfigure(1, weight=1)
        pass
    def showIsDebugingValue(self):
        self.status_var.set(f"Debugging is {self.isDebugging.get()}")
            
    def setDebuggingSliders(self, direction, value):
        self.DebuggingValue[direction] = value
        
    def goto_target_callback(self):
        if self.current_Mode.get() == "SimIK":
            simIK.setGroupCalculation(self.IKEnvorement,self.IKGroup,simIK.method_damped_least_squares,0.1,100)
            options = {
                'syncWorlds': True,
                'allowError': True
            }
            result,flags,precision=simIK.handleGroup(self.IKEnvorement,self.IKGroup,options)
            
            print(result == simIK.result_success)
            print(flags)
            print(precision)
            for i in range(6):
                pos = sim.getJointTargetPosition(self.joint_handles[i])
                posDegree = math.degrees(pos)
                self.jointslidersVar[i].set(posDegree)
            return
        target_position = sim.getObjectPosition(target, -1)
        x_real, y_real = self.DebuggingValue["x"], self.DebuggingValue["y"]
        if not self.isDebugging.get():
            x_real, y_real = target_position[0], target_position[1]
        
        # Convert to meters (CoppeliaSim uses meters)
        x_m = x_real
        y_m = y_real
        z_m = 0.1  # Fixed height above table (10cm)
        
        # Calculate inverse kinematics
        joint_angles = inverse_kinematics([x_m, y_m, z_m])
        
        if joint_angles is None:
            messagebox.showerror("Error", "Target position unreachable")
            return
            
        # Set joint angles in CoppeliaSim
        for i in range(6):
            print(f"joint {i} : {joint_angles[i]}")
            sim.setJointTargetPosition(self.joint_handles[i], joint_angles[i])
            
        self.status_var.set(f"Moving to target at ({x_real:.1f}, {y_real:.1f}) cm")  
        
    # Set target positions in CoppeliaSim
    def slider_moved(self, joint_index, degrees):
        print("Slider moved:", joint_index, "Value:", degrees)
        if joint_index < len(self.joint_handles) and self.joint_handles[joint_index]:
            radians = math.radians(degrees)
            sim.setJointTargetPosition(self.joint_handles[joint_index], radians)

    def create_widgets(self):
        # Warping controls
        for i in range(8):
            self.control_frame.grid_columnconfigure(i, weight=1)
        self.control_frame.grid_rowconfigure(0, weight=1)
            
        warp_frame = tk.LabelFrame(self.control_frame, text="Image Warping", padx=5, pady=5, relief="groove")
        warp_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        tk.Button(warp_frame, text="Select Corners", command=self.enable_corner_selection, width=15).pack(side="top", padx=5)
        tk.Button(warp_frame, text="Reset Corners", command=self.reset_corners, width=15).pack(side="top", padx=5)
        
        # Pause/Resume button
        control_frame = tk.LabelFrame(self.control_frame, text="Controls", padx=5, pady=5)
        control_frame.grid(row=0, column=1, sticky="nsew", pady=5)
        
        self.pause_btn = tk.Button(control_frame, text="Pause", command=self.toggle_pause, width=15)
        self.pause_btn.pack(side="top", padx=5)
        
        # Calibration controls
        cal_frame = tk.LabelFrame(self.control_frame, text="Calibration", padx=5, pady=5)
        cal_frame.grid(row=0, column=2, sticky="nsew", pady=5)
        
        tk.Button(cal_frame, text="Toggle Calibration", command=self.toggle_calibration, width=18).pack(side="top", padx=5)
        tk.Button(cal_frame, text="Add Point", command=self.add_calibration_point, width=18).pack(side="top", padx=5)
        tk.Button(cal_frame, text="Calculate Regression", command=self.calculate_regression, width=18).pack(side="top", padx=5)
        tk.Button(cal_frame, text="Save Calibration", command=self.save_calibration_data, width=18).pack(side="top", padx=5)
        
        # Calibration data display
        data_frame = tk.LabelFrame(self.control_frame, text="Calibration Data", padx=5, pady=5)
        data_frame.grid(row=0, column=3, sticky="nsew", pady=5)
        
        self.data_text = tk.Text(data_frame, height=6, width=50)
        self.data_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.data_text.insert("end", "No calibration data\n")
        self.data_text.config(state="disabled")
        
        # Position display
        pos_frame = tk.LabelFrame(self.control_frame, text="Position Information", padx=5, pady=5)
        pos_frame.grid(row=0, column=4, sticky="nsew", pady=5)
        
        tk.Label(pos_frame, text="Pixel Position:").grid(row=0, column=0, sticky="w")
        self.pixel_pos_var = tk.StringVar(value="N/A")
        tk.Label(pos_frame, textvariable=self.pixel_pos_var).grid(row=0, column=1, sticky="w")
        
        tk.Label(pos_frame, text="Real Position:").grid(row=1, column=0, sticky="w")
        self.real_pos_var = tk.StringVar(value="N/A")
        tk.Label(pos_frame, textvariable=self.real_pos_var).grid(row=1, column=1, sticky="w")
        
        # Regression parameters
        reg_frame = tk.LabelFrame(self.control_frame, text="Regression Parameters", padx=5, pady=5)
        reg_frame.grid(row=0, column=5, sticky="nsew", pady=5)
        
        tk.Label(reg_frame, text="X: real = a1 + b1 * pixel").grid(row=0, column=0, sticky="w")
        self.reg_x_var = tk.StringVar(value="Not calculated")
        tk.Label(reg_frame, textvariable=self.reg_x_var).grid(row=0, column=1, sticky="w")
        
        tk.Label(reg_frame, text="Y: real = a2 + b2 * pixel").grid(row=1, column=0, sticky="w")
        self.reg_y_var = tk.StringVar(value="Not calculated")
        tk.Label(reg_frame, textvariable=self.reg_y_var).grid(row=1, column=1, sticky="w")
        
        # Mode indicators
        mode_frame = tk.Frame(self.control_frame)
        mode_frame.grid(row=0, column=6, sticky="nsew", pady=5)
        
        self.cal_mode_var = tk.StringVar(value="Calibration: OFF")
        tk.Label(mode_frame, textvariable=self.cal_mode_var, fg="red").pack(side="top", padx=5)
        
        self.warp_mode_var = tk.StringVar(value="Warping: OFF")
        tk.Label(mode_frame, textvariable=self.warp_mode_var, fg="red").pack(side="top", padx=5)
        
        self.pause_mode_var = tk.StringVar(value="Paused: NO")
        tk.Label(mode_frame, textvariable=self.pause_mode_var, fg="red").pack(side="top", padx=5)

    def update_video_feed(self):
        global pause, warp_matrix, corners, is_corner_ready, calibration_mode
        
        if not pause:
            try:
                # Get the latest frame from the queue
                frame = frame_queue.get_nowait()
                self.current_frame = frame
                
                # Create a copy for display annotations
                display_frame = frame.copy()
                
                # Apply warping if ready
                if warp_matrix is not None:
                    # Warp the frame
                    frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                    warped = cv2.warpPerspective(frame_bgr, warp_matrix, warp_size)
                    self.warped_frame = warped.copy()
                    
                    # Convert back to RGB for display
                    display_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    self.warp_mode_var.set("Warping: ON")
                    
                    # Run detection if model is available
                    if model is not None:
                        results = model(warped)
                        # display_frame = results[0].plot()
                        self.current_detections = []
                        
                        if results and results[0].boxes.xyxy is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            for box in boxes:
                                x1, y1, x2, y2 = box
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                self.current_detections.append((center_x, center_y))
                                
                                # Draw detection on display frame
                                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                                
                                # Convert to real-world coordinates
                                if a1 is not None and b1 is not None:
                                    x_real, y_real = self.pixel_to_real(center_x, center_y)
                                    
                                    # Update CoppeliaSim
                                    if sim.getSimulationState() == sim.simulation_advancing_running:
                                        targetPosition = sim.getObjectPosition(target, -1)
                                        sim.setObjectPosition(target, -1, [x_real / 40.0, y_real / 40.0, targetPosition[2]])
                                    
                                    # Display coordinates
                                    coord_text = f"({x_real:.1f}, {y_real:.1f})"
                                    cv2.putText(display_frame, coord_text, 
                                               (center_x + 20, center_y - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                               (0, 255, 0), 2)
                                    
                                    # Update position display
                                    self.pixel_pos_var.set(f"({center_x}, {center_y})")
                                    self.real_pos_var.set(f"({x_real:.2f}, {y_real:.2f})")
                                else:
                                    self.pixel_pos_var.set(f"({center_x}, {center_y})")
                                    self.real_pos_var.set("Calibration needed")
                else:
                    self.warp_mode_var.set("Warping: OFF")
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
                
                # Show calibration status
                if calibration_mode:
                    cv2.putText(display_frame, "CALIBRATION MODE", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Points: {len(calibration_points)}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Convert to PhotoImage and update label
                img = Image.fromarray(display_frame)
                img_tk = ImageTk.PhotoImage(image=img)
                self.image_label.img_tk = img_tk  # Keep reference
                self.image_label.config(image=img_tk)
                
            except queue.Empty:
                # No new frame available, skip this update
                pass
        
        # Schedule next update
        self.root.after(30, self.update_video_feed)

    def enable_corner_selection(self):
        global is_corner_ready, corners
        corners = []
        is_corner_ready = False
        self.status_var.set("Click on the image to select 4 corners (TL, TR, BR, BL)")

    def reset_corners(self):
        global corners, warp_matrix
        corners = []
        warp_matrix = None
        self.status_var.set("Corners reset")

    def toggle_pause(self):
        global pause
        pause = not pause
        self.pause_mode_var.set(f"Paused: {'YES' if pause else 'NO'}")
        self.pause_btn.config(text="Resume" if pause else "Pause")
        self.status_var.set(f"System {'paused' if pause else 'resumed'}")

    def toggle_calibration(self):
        global calibration_mode
        calibration_mode = not calibration_mode
        self.cal_mode_var.set(f"Calibration: {'ON' if calibration_mode else 'OFF'}")
        self.status_var.set(f"Calibration mode {'ON' if calibration_mode else 'OFF'}")

    def add_calibration_point(self):
        global calibration_points, current_object_pos
        
        if not self.current_detections:
            messagebox.showerror("Error", "No object detected. Please detect an object first.")
            return
        
        # Use the first detected object
        px, py = self.current_detections[0]
        
        # Get real-world coordinates
        real_x = simpledialog.askfloat("Real X", "Enter real-world X coordinate (cm):", parent=self.root)
        if real_x is None: return
        
        real_y = simpledialog.askfloat("Real Y", "Enter real-world Y coordinate (cm):", parent=self.root)
        if real_y is None: return
        
        # Add to calibration points
        calibration_points.append((px, py, real_x, real_y))
        
        # Update data display
        self.update_calibration_display()
        self.status_var.set(f"Added calibration point: Pixel ({px}, {py}) -> Real ({real_x}, {real_y})")

    def update_calibration_display(self):
        self.data_text.config(state="normal")
        self.data_text.delete(1.0, "end")
        
        if not calibration_points:
            self.data_text.insert("end", "No calibration data\n")
        else:
            self.data_text.insert("end", "Pixel X\tPixel Y\tReal X\tReal Y\n")
            self.data_text.insert("end", "-"*40 + "\n")
            for px, py, rx, ry in calibration_points:
                self.data_text.insert("end", f"{px:.1f}\t{py:.1f}\t{rx:.2f}\t{ry:.2f}\n")
        
        self.data_text.config(state="disabled")

    def calculate_regression(self):
        global a1, b1, a2, b2, calibration_points
        
        if len(calibration_points) < 2:
            messagebox.showerror("Error", "Need at least 2 calibration points")
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
        
        # Update display
        self.reg_x_var.set(f"a1 = {a1:.6f}, b1 = {b1:.6f}")
        self.reg_y_var.set(f"a2 = {a2:.6f}, b2 = {b2:.6f}")
        self.status_var.set("Regression calculated successfully")

    def pixel_to_real(self, x_pixel, y_pixel):
        if a1 is None or b1 is None or a2 is None or b2 is None:
            return None, None
        x_real = a1 + b1 * x_pixel
        y_real = a2 + b2 * y_pixel
        return x_real, y_real

    def save_calibration_data(self):
        try:
            with open(CALIBRATION_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['pixel_x', 'pixel_y', 'real_x', 'real_y'])
                for point in calibration_points:
                    writer.writerow(point)
                # Write regression parameters
                writer.writerow(['Regression Parameters'])
                writer.writerow(['a1', a1])
                writer.writerow(['b1', b1])
                writer.writerow(['a2', a2])
                writer.writerow(['b2', b2])
            self.status_var.set(f"Calibration data saved to {CALIBRATION_FILE}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving calibration data: {e}")

    def load_calibration_data(self):
        global calibration_points, a1, b1, a2, b2
        
        if not os.path.exists(CALIBRATION_FILE):
            return
            
        try:
            calibration_points = []
            with open(CALIBRATION_FILE, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if not row:
                        continue
                    if row[0] == 'Regression Parameters':
                        # Read parameters
                        a1 = float(next(reader)[1])
                        b1 = float(next(reader)[1])
                        a2 = float(next(reader)[1])
                        b2 = float(next(reader)[1])
                        break
                    elif len(row) >= 4:
                        try:
                            px, py, rx, ry = map(float, row[:4])
                            calibration_points.append((px, py, rx, ry))
                        except ValueError:
                            continue
            
            self.update_calibration_display()
            
            # Update regression display if parameters loaded
            if a1 is not None and b1 is not None:
                self.reg_x_var.set(f"a1 = {a1:.6f}, b1 = {b1:.6f}")
                self.reg_y_var.set(f"a2 = {a2:.6f}, b2 = {b2:.6f}")
            
            self.status_var.set(f"Loaded {len(calibration_points)} calibration points")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading calibration data: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDetectionAPP(root)
    root.mainloop()