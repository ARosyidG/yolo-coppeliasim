
import math


print(math.degrees(math.pi/2))


sim = 1
target = 3

# Add this function to your script (inside the ImageDetectionAPP class)
def inverse_kinematics(self, x, y, z):
    """
    Calculate joint angles for UR5 robot using analytical inverse kinematics
    based on DH parameters. Returns angles in radians.
    """
    # UR5 DH parameters (converted to meters)
    d1 = 0.0661
    a2 = 0.4251
    a3 = 0.3922
    d4 = 0.0397
    d5 = 0.0492
    d6 = 0.0804
    
    # Calculate wrist center position
    # Assuming end effector points down (z-direction)
    wrist_z = z + d5 + d6
    
    # Calculate theta1 (base rotation)
    theta1 = math.atan2(y, x)
    
    # Calculate radial distance from base to wrist center
    r = math.sqrt(x**2 + y**2)
    
    # Calculate height relative to joint1
    h = wrist_z - d1
    
    # Calculate distance in the arm plane
    D = math.sqrt(r**2 + h**2)
    
    # Check if position is reachable
    if D > (a2 + a3) or D < abs(a2 - a3):
        return None  # Unreachable position
    
    # Calculate angles using law of cosines
    # For theta3
    cos_theta3 = (D**2 - a2**2 - a3**2) / (2 * a2 * a3)
    sin_theta3 = math.sqrt(1 - cos_theta3**2)  # Elbow up solution
    
    theta3 = math.atan2(sin_theta3, cos_theta3)
    
    # For theta2
    alpha = math.atan2(h, r)
    beta = math.atan2(a3 * sin_theta3, a2 + a3 * cos_theta3)
    theta2 = alpha + beta
    
    # For last three joints (simplified for wrist orientation)
    # Assuming fixed orientation with end effector pointing down
    theta4 = -theta2 - theta3  # Keep wrist horizontal
    theta5 = -math.pi/2        # Point down
    theta6 = 0                 # Neutral rotation
    
    return [theta1, theta2, theta3, theta4, theta5, theta6]

# Modify the goto_target_callback function
def goto_target_callback(self):
    if self.current_Mode.get() == "SimIK":
        # Existing SimIK code...
        pass
    elif self.current_Mode.get() == "Manual":
        # Get target position from CoppeliaSim
        target_pos = sim.getObjectPosition(target, -1)
        
        # Apply debugging offsets if in debugging mode
        if self.isDebugging.get():
            target_pos[0] += self.DebuggingValue["x"]
            target_pos[1] += self.DebuggingValue["y"]
        
        # Convert to meters (assuming simulation uses meters)
        x, y, z = target_pos
        
        # Calculate inverse kinematics
        joint_angles = self.inverse_kinematics(x, y, z)
        
        if joint_angles is None:
            self.status_var.set("Position unreachable!")
            return
        
        # Apply DH offsets
        offsets = [math.pi/2, 0, 0, -math.pi/2, math.pi/2, 0]
        final_angles = [angle - offset for angle, offset in zip(joint_angles, offsets)]
        
        # Convert to degrees and update sliders
        for i in range(6):
            degrees = math.degrees(final_angles[i])
            self.jointslidersVar[i].set(degrees)
        
        self.status_var.set("Manual IK solution applied")