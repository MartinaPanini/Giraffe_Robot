# -*- coding: utf-8 -*-
"""
Configuration for task-space control of giraffe robot
"""

import numpy as np
import pinocchio as pin

# Controller parameters
dt = 0.001                     # controller time step [s]
exp_duration = 10.0            # simulation duration [s]
SLOW_FACTOR = 1                # real-time factor
frame_name = 'ee_link'         # controlled frame

# Task-space PD gains
Kp_task = np.diag([400., 400., 400.])  # Position gain
Kd_task = np.diag([40., 40., 40.])     # Velocity gain

# --- ADDED FOR ORIENTATION CONTROL ---
# Orientation task gains
Kp_ori = np.diag([100.0, 100.0, 100.0]) # Orientation gain
Kd_ori = np.diag([10.0, 10.0, 10.0])    # Orientation damping gain
# ------------------------------------

# Postural task gains
Kp_postural = 10.0                  # Postural position gain
Kd_postural = 2.0                   # Postural velocity gain

# Initial and desired configurations
qhome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Homing configuration
q0 = np.array([0.0, -np.pi/2, 0.0, 0.0, -np.pi/2])  # Reference posture
pdes = np.array([1.0, 2.0, 1.0])  # Desired position [x,y,z]

# Trajectory parameters
traj_duration = 3.0            # Time to reach target

# Desired orientation (pitch angle) from the image
# This is no longer used directly by task_space.py, but is good for reference
theta_des = np.radians(-30.0)
axis_des = np.array([0, 1, 0])  # Asse di rotazione (Y for pitch)

# Aggiungi questa riga alla fine del file
print("--- task_conf.py caricato correttamente ---")