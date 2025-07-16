import numpy as np
import os
import math

# Initial Conditions
q0 = np.array([0.0, -np.pi/3, np.pi/4, -np.pi/3, 0.0])         # position
qd0 =  np.array([0.0, -5.0, 0.5, 0.0, 0.0])                    # velocity
qdd0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # accelerations

# Controller parameters
dt = 0.001                     # Controller time step [s]
exp_duration = 6.0             # Simulation duration [s]
exp_dyn_duration = 3.0
SLOW_FACTOR = 5                # Real-time factor
frame_name = 'ee_link'         # Controlled frame

# Position gains
Kp_pos = np.diag([100., 100., 100.])
Kd_pos = np.diag([10., 10., 10.])

# Orientation gains
Kp_pitch = 10.0
Kd_pitch = 2.0

# Postural gains
Kp_postural = 10.0
Kd_postural = 10.0

# Initial joint configuratio
qhome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# End-Effector desired position [x, y, z]
pdes = np.array([1.0, 2.0, 1.0])

# Pitch orientation desired
pitch_des_deg = -30.0

# Trajectory duration to reach the target
traj_duration = 3.0  