import math
import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# UR5 DH Parameters
d1, a1, alpha1 = 0.0661, 0.0000, -math.pi/2
d2, a2, alpha2 = 0.0000, 0.4251, 0
d3, a3, alpha3 = 0.0000, 0.3922, 0
d4, a4, alpha4 = 0.0397, 0.0000, -math.pi/2
d5, a5, alpha5 = 0.0492, 0.0000, -math.pi/2
d6, a6, alpha6 = 0.0000, 0.0000, 0

def dh_transform(theta, d, a, alpha):
    """Compute a single DH transformation matrix"""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q):
    """Compute UR5 end effector pose from joint angles"""
    T01 = dh_transform(q[0], d1, a1, alpha1)
    T12 = dh_transform(q[1], d2, a2, alpha2)
    T23 = dh_transform(q[2], d3, a3, alpha3)
    T34 = dh_transform(q[3], d4, a4, alpha4)
    T45 = dh_transform(q[4], d5, a5, alpha5)
    T56 = dh_transform(q[5], d6, a6, alpha6)
    
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T06

def get_wrist_center(ee_pos, R0e, d6=0.0):
    """Compute wrist center position"""
    # Vector from wrist to EE in base frame (z-direction of EE frame)
    wrist_to_ee = R0e[:, 2] * d6
    return ee_pos - wrist_to_ee

def solve_first_three_joints(wrist_center):
    """Analytical solution for q1, q2, q3"""
    x, y, z = wrist_center
    
    # Solve for q1 (two solutions)
    q1_a = math.atan2(y, x)
    q1_b = math.atan2(-y, -x)
    
    # Solve for q2, q3 (for both q1 solutions)
    solutions = []
    for q1 in [q1_a, q1_b]:
        # Transform wrist center to shoulder frame
        x_shoulder = x * math.cos(q1) + y * math.sin(q1) - a1
        z_shoulder = z - d1
        
        # Distance from shoulder to wrist center projection
        D = math.sqrt(x_shoulder**2 + z_shoulder**2)
        
        # Solve triangle using law of cosines
        try:
            # Elbow-up configuration
            gamma = math.atan2(z_shoulder, x_shoulder)
            beta = math.acos((a2**2 + D**2 - a3**2) / (2 * a2 * D))
            q2_up = math.pi/2 - gamma - beta
            
            # Elbow-down configuration
            q2_down = math.pi/2 - gamma + beta
            
            # Corresponding q3 solutions
            q3_up = math.acos((a2**2 + a3**2 - D**2) / (2 * a2 * a3)) - math.pi
            q3_down = -q3_up
            
            solutions.append((q1, q2_up, q3_up))
            solutions.append((q1, q2_down, q3_down))
        except:
            # Solution not possible for this configuration
            continue
    
    return solutions

def solve_last_three_joints(q1, q2, q3, R0e):
    """Solve for wrist joints using orientation"""
    # Compute rotation from base to joint 3
    T01 = dh_transform(q1, d1, a1, alpha1)
    T12 = dh_transform(q2, d2, a2, alpha2)
    T23 = dh_transform(q3, d3, a3, alpha3)
    T03 = T01 @ T12 @ T23
    
    # Extract rotation components
    R03 = T03[:3, :3]
    R36 = np.linalg.inv(R03) @ R0e
    
    # Extract Euler angles from R36 (ZYZ convention)
    # Handle singularity when sin(q5) ≈ 0
    if abs(R36[2, 2]) > 0.9999:
        q5 = 0.0
        q4 = math.atan2(R36[1, 0], R36[0, 0])
        q6 = 0.0
    else:
        q4 = math.atan2(R36[1, 2], R36[0, 2])
        q5 = math.atan2(math.sqrt(R36[0, 2]**2 + R36[1, 2]**2), R36[2, 2])
        q6 = math.atan2(R36[2, 1], -R36[2, 0])
    
    return q4, q5, q6

def inverse_kinematics(x, y, z, roll, pitch, yaw):
    """Compute joint angles for desired EE pose"""
    # Convert orientation to rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    ee_pos = np.array([x, y, z])
    
    # Compute wrist center position
    wrist_center = get_wrist_center(ee_pos, R, d6=0.0)
    
    # Solve for first three joints (multiple solutions)
    solutions = solve_first_three_joints(wrist_center)
    
    # Solve for wrist joints for each solution
    complete_solutions = []
    for q1, q2, q3 in solutions:
        try:
            q4, q5, q6 = solve_last_three_joints(q1, q2, q3, R)
            complete_solutions.append((q1, q2, q3, q4, q5, q6))
        except:
            continue
    
    return complete_solutions

# Helper functions
def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix (ZYX convention)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def plot_arm(q):
    """Visualize UR5 configuration"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute joint positions
    positions = [np.array([0, 0, 0])]
    T = np.eye(4)
    
    for i, angle in enumerate(q):
        T = T @ dh_transform(angle, 
                           [d1, d2, d3, d4, d5, d6][i],
                           [a1, a2, a3, a4, a5, a6][i],
                           [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6][i])
        positions.append(T[:3, 3])
    
    # Plot
    pos = np.array(positions)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.5])
    plt.show()

# Example usage
if __name__ == "__main__":
    # Desired end effector pose
    x, y, z = 0.12152, 0.55073, 0.05
    roll, pitch, yaw = 0, math.pi/2, 0
    
    # Compute IK solutions
    solutions = inverse_kinematics(x, y, z, roll, pitch, yaw)
    
    if solutions:
        print(f"Found {len(solutions)} solutions:")
        for i, q in enumerate(solutions):
            print(f"Solution {i+1}: {[round(ang, 3) for ang in q]}")
            
            # Verify solution
            T = forward_kinematics(q)
            print("Achieved position:", np.round(T[:3, 3], 4))
            
            # Visualize
            plot_arm(q)
    else:
        print("No valid IK solutions found")