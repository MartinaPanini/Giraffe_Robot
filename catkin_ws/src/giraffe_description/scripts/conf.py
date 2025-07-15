import numpy as np
import os
import math

# Initial Conditions
q0 = np.array([0.0, -np.pi/3, np.pi/4, -np.pi/3, 0.0])         # position
qd0 =  np.array([0.0, -5.0, 0.5, 0.0, 0.0])                    # velocity
qdd0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # accelerations

# Controller parameters
dt = 0.001                     # Controller time step [s]
exp_duration = 6.0            # Simulation duration [s]
SLOW_FACTOR = 5                # Real-time factor
frame_name = 'ee_link'         # Controlled frame

# Guadagni per la parte di posizione (3D) del task
Kp_pos = np.diag([100., 100., 100.])
Kd_pos = np.diag([10., 10., 10.])

# Guadagni per la parte di orientamento (1D - Pitch) del task
Kp_pitch = 10.0
Kd_pitch = 2.0

# Guadagni per il task secondario di postura (ridondanza)
Kp_postural = 10.0
Kd_postural = 10.0

# Configurazione iniziale dei giunti all'avvio
qhome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Posizione desiderata dell'end-effector [x, y, z]
pdes = np.array([1.0, 2.0, 1.0])

# Orientamento di pitch desiderato in gradi
pitch_des_deg = -30.0

# Parametri della traiettoria
traj_duration = 3.0  # Durata in secondi per raggiungere il target