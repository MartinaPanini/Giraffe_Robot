#!/usr/bin/env python
from __future__ import print_function

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.in_kin_pinocchio import robotKinematics
from utils.math_tools import Math

from kinematics import *
from task_space import simulation

import conf as conf

#os.system("killall rosmaster rviz")
#instantiate graphic utils
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
data = robot.data
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

#####################################################################
# Test Simulation
#####################################################################

# Variables initialization
q = conf.qhome.copy()
qd = np.zeros(robot.nv)
qdd = np.zeros(robot.nv)

pitch_des_final = np.radians(conf.pitch_des_deg)

simulation(pitch_des_final, q, qd)
# Stampa la posa finale
final_pos = data.oMf[frame_id].translation
final_orient_rpy = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
print("\nPosizione Finale dell'End Effector (m):", final_pos)
print("Orientamento Finale dell'End Effector (RPY - deg):", np.degrees(final_orient_rpy))
print("Pitch Finale (deg):", np.degrees(final_orient_rpy[1]))
print("Pitch Desiderato (deg):", np.degrees(pitch_des_final))
