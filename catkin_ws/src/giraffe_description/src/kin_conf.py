import numpy as np
import os
import math

dt = 0.001                   # controller time step
SLOW_FACTOR = 10             # to slow down simulation
frame_name = 'ee_link'       # name of the frame to control (end-effector) in the URDF

# Initial Conditions
q0 =   np.array([0.0, -math.pi, 0.0, 0.0, -math.pi/2])        # position
qd0 =  np.array([0.0, -5.0, 0.5, 0.0, 0.0])                    # velocity
qdd0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                    # accelerations





