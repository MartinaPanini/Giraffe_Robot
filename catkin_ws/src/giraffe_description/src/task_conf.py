# -*- coding: utf-8 -*-
"""
Configuration for 4D task-space control of the giraffe robot.
The primary task is 3D position + 1D pitch control.
"""

import numpy as np

# Controller parameters
dt = 0.001                     # Controller time step [s]
exp_duration = 10.0            # Simulation duration [s]
SLOW_FACTOR = 1.5                # Real-time factor
frame_name = 'ee_link'         # Controlled frame

# --------------------------------------------------------------------
# --- GUADAGNI PER IL CONTROLLO 4D ---
# --------------------------------------------------------------------

# Guadagni per la parte di posizione (3D) del task
Kp_pos = np.diag([100., 100., 100.])
Kd_pos = np.diag([10., 10., 10.])

# Guadagni per la parte di orientamento (1D - Pitch) del task
Kp_pitch = 10.0
Kd_pitch = 2.0

# Guadagni per il task secondario di postura (ridondanza)
# Vengono usati per guidare dolcemente il robot verso una configurazione
# coerente, calcolata all'inizio dello script principale.
Kp_postural = 10.0
Kd_postural = 10.0

# --------------------------------------------------------------------
# --- CONFIGURAZIONI INIZIALI E DESIDERATE ---
# --------------------------------------------------------------------

# Configurazione iniziale dei giunti all'avvio
qhome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Postura di riferimento.
# NOTA: Questa è solo una configurazione iniziale o di "ripiego".
# Lo script principale calcolerà un obiettivo posturale migliore ('q0_calibrated')
# che sia cinematicamente coerente con la posa finale desiderata.
q0 = np.array([0.0, -np.pi/3, np.pi/4, -np.pi/3, 0.0])

# Posizione desiderata dell'end-effector [x, y, z]
# NOTA: Questo è un punto di esempio. L'assignment richiede che il robot
# possa raggiungere qualsiasi punto in un'area di 5x12 metri.
pdes = np.array([1.0, 2.0, 1.0])

# Orientamento di pitch desiderato in gradi
# (Negativo per puntare verso il basso, come nel diagramma della sedia)
pitch_des_deg = -30.0

# Parametri della traiettoria
traj_duration = 3.0  # Durata in secondi per raggiungere il target