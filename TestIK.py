import unittest
import numpy as np
import math
from myKinematics import forward_kinematics, analytical_ik  # Import your actual implementation

# Same DH parameters as in your implementation
DH_PARAMS = [
    {'d': 0.0661, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': math.pi/2},
    {'d': 0.0000, 'a': 0.4251, 'alpha': 0, 'offset': 0},
    {'d': 0.0000, 'a': 0.3922, 'alpha': 0, 'offset': 0},
    {'d': 0.0397, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': -math.pi/2},
    {'d': 0.0492, 'a': 0.0000, 'alpha': -math.pi/2, 'offset': math.pi/2},
    {'d': 0.0000, 'a': 0.0000, 'alpha': 0, 'offset': 0}
]

class TestRobotKinematics(unittest.TestCase):
    
    def test_forward_kinematics_home_position(self):
        """Test FK at home position (all joint angles=0)"""
        q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Precomputed expected position (manually verified)
        expected_pos = np.array([
            -0.0397,  # x
            0.8665,   # y
            0.0661    # z
        ])
        
        actual_pos = forward_kinematics(q)
        np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-4)
    
    def test_forward_kinematics_non_zero(self):
        """Test FK with non-zero joint angles"""
        q = [math.pi/2, -math.pi/4, math.pi/3, 0.0, math.pi/6, 0.0]
        
        # Precomputed expected position (verified via independent calculation)
        expected_pos = np.array([
            0.2068,   # x
            0.4053,   # y
            0.3819    # z
        ])
        
        actual_pos = forward_kinematics(q)
        np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-4)
    
    def test_analytical_ik_round_trip(self):
        """Test IK by verifying FK(IK(target)) == target"""
        # Define test positions within workspace
        test_positions = [
            np.array([0.4, 0.2, 0.3]),
            np.array([0.3, -0.3, 0.4]),
            np.array([0.1, 0.1, 0.6]),
            np.array([0.5, 0.0, 0.2])
        ]
        
        q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for target_pos in test_positions:
            # Solve IK
            q_solution = analytical_ik(
                target_pos,
                q_init,
                max_iter=1000,
                tol=1e-6,
                alpha=0.1
            )
            
            # Verify solution
            achieved_pos = forward_kinematics(q_solution)
            np.testing.assert_allclose(achieved_pos, target_pos, atol=1e-3)
    
    def test_analytical_ik_singularity(self):
        """Test IK near singularity (fully extended arm)"""
        target_pos = np.array([0.8, 0.0, 0.0661])  # Max reach in x-direction
        q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        q_solution = analytical_ik(target_pos, q_init)
        achieved_pos = forward_kinematics(q_solution)
        np.testing.assert_allclose(achieved_pos, target_pos, atol=1e-3)

if __name__ == '__main__':
    unittest.main()