#!/usr/bin/env python
from __future__ import print_function

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.in_kin_pinocchio import robotKinematics
from utils.math_tools import Math

from kinematics import *
from task_space import simulation
from dynamics import dyn_simulation

import conf as conf

# Inizializza RosPub una sola volta all'inizio
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
data = robot.data
model = robot.model
kin = robotKinematics(robot, conf.frame_name)

# Init variables
zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
time = 0.0

q = conf.q0.copy()
qd = conf.qd0.copy()
qdd = conf.qdd0.copy()

q_des = conf.q0
qd_des = conf.qd0
qdd_des = conf.qdd0

math_utils = Math()

# get the ID corresponding to the frame we want to control
assert(robot.model.existFrame(conf.frame_name))
frame_id = robot.model.getFrameId(conf.frame_name)

#####################################################################
# Test Direct Kinematics
#####################################################################
#dk_test(q, qd, robot, frame_id)
#jacobian_test(q, frame_id, robot)

#####################################################################
# Test Dynamics
#####################################################################
assert(robot.model.existFrame(conf.frame_name))
frame_ee = robot.model.getFrameId(conf.frame_name)
dyn_simulation(robot, time, ros_pub, q, qd, qdd, q_des, qd_des, qdd_des)

#####################################################################
# Test Simulation
#####################################################################

# # Variables initialization
# q_sim = conf.qhome.copy() # Usa variabili separate per la simulazione
# qd_sim = np.zeros(robot.nv) # Usa variabili separate per la simulazione

# pitch_des_final = np.radians(conf.pitch_des_deg)

# # Cattura i valori finali di q e qd restituiti dalla simulazione
# q_final, qd_final = simulation(ros_pub, robot, model, data, pitch_des_final, q_sim, qd_sim)

# # Aggiorna esplicitamente la cinematica diretta del robot con i valori finali
# pin.forwardKinematics(model, data, q_final, qd_final)
# pin.updateFramePlacement(model, data, frame_id)

# # Stampa la posa finale usando i dati aggiornati
# final_pos = data.oMf[frame_id].translation
# final_orient_rpy = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
# print("\nPosizione Finale dell'End Effector (m):", final_pos)
# print("Orientamento Finale dell'End Effector (RPY - deg):", np.degrees(final_orient_rpy))
# print("Pitch Finale (deg):", np.degrees(final_orient_rpy[1]))
# print("Pitch Desiderato (deg):", np.degrees(pitch_des_final))

ros_pub.deregister_node()