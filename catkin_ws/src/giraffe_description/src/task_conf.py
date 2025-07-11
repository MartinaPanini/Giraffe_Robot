# -*- coding: utf-8 -*-
"""
Configuration for task-space control of giraffe robot
"""

import numpy as np

# Controller parameters
dt = 0.001                     # controller time step [s]
exp_duration = 10.0            # simulation duration [s]
SLOW_FACTOR = 1                # real-time factor
frame_name = 'ee_link'         # controlled frame

# Task-space PD gains (Requirement 6)
omega_n = 4/7                  # Natural frequency for 7s settling time (critically damped)
# Example good starting values
Kp_task = np.diag([400, 400, 400])  # Position gain
Kd_task = np.diag([40, 40, 40])     # Velocity gain

# Postural task gains (Requirement 7)
Kp_postural = 10.0                  # Postural position gain
Kd_postural = 2.0                   # Postural velocity gain

# Initial and desired configurations (Requirement 8)
qhome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Homing configuration
q0 = np.array([0.0, -np.pi/2, 0.0, 0.0, -np.pi/2])  # Reference posture
pdes = np.array([1.0, 2.0, 1.0])  # Desired position [x,y,z]

# Trajectory parameters
traj_duration = 3.0            # Time to reach target

# Orientamento desiderato (30 gradi sull'asse Z)
theta_des = np.radians(30)  # Conversione da gradi a radianti
axis_des = np.array([0, 0, 1])  # Asse di rotazione (Z)