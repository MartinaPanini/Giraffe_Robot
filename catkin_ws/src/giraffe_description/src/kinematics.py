#!/usr/bin/env python
# coding=utf-8 
from __future__ import print_function

import pinocchio as pin
from pinocchio.utils import *
import numpy as np
from numpy import nan
import math
import time as tm

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.kin_dyn_utils import directKinematics
from utils.kin_dyn_utils import computeEndEffectorJacobian
from utils.kin_dyn_utils import numericalInverseKinematics as ik
from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj
from utils.kin_dyn_utils import geometric2analyticJacobian
from utils.in_kin_pinocchio import robotKinematics
from utils.math_tools import Math
import matplotlib.pyplot as plt
from utils.common_functions import plotJoint

import conf as conf

#os.system("killall rosmaster rviz")
#instantiate graphic utils
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
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

#################
# Direct kinematics
#################
# direct kinematics function
T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt = directKinematics(q)
# compare with Pinocchio built-in functions 
robot.computeAllTerms(q, qd)
x = robot.framePlacement(q, frame_id).translation
o = robot.framePlacement(q, frame_id).rotation
position_diff = x - T_wt[:3,3]
rotation_diff = o - T_wt[:3,:3]
print("Direct Kinematics - ee position, differece with Pinocchio library:", position_diff)
print("Direct Kinematics - ee orientation, differece with Pinocchio library:\n", rotation_diff)
print("------------------------------------------")
print("Manual DK position:", T_wt[:3, 3])
print("Pinocchio position:", x)
print("Difference:", T_wt[:3, 3] - x)

#################
# Geometric Jacobian
#################
J, z1, z2, z3, z4, z5 = computeEndEffectorJacobian(q)
# compare with Pinocchio
Jee = robot.frameJacobian(q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
jacobian_diff = J - Jee
print("\n------------------------------------------")
np.set_printoptions(suppress=True, precision=2)
print("Geometric Jacobian:\n", J)
print("Direct Kinematics - ee Gometric Jacobian (6X5 matrix), differece with Pinocchio library:\n", jacobian_diff)

#################
# Analytic Jacobian
#################
J_a = geometric2analyticJacobian(J, T_wt)
print("\n------------------------------------------")
print("Analytic Jacobian:\n", J_a)
ros_pub.deregister_node()
#plt.show(block=True)

##################
# Inverse Kinematics (Numerical)
##################

# desired task space position
p = np.array([1,2,1])
p_d = np.array([1,2,1,0])

# initial guess
q_i  = np.array([0.0, -math.pi, 0.0, 0.0, -math.pi/2])

q_postural = np.array([math.pi, 0.0, -math.pi, 0.0, 0.0])

# solution of the numerical ik
q_f, log_err, log_grad = ik(p_d, q_i, line_search = True, wrap = True)

# sanity check
# compare solution with values obtained through direct kinematics
T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt = directKinematics(q_f)
rpy = math_utils.rot2eul(T_wt[:3,:3])
task_diff = p_d - np.hstack((T_wt[:3,3],rpy[0]))

print("\n------------------------------------------")
print("Desired End effector \n", p)
print("Point obtained with IK solution \n", np.hstack((T_wt[:3, 3], rpy[0])))
print("Norm of error at the end-effector position: \n", np.linalg.norm(task_diff))
print("Final joint positions\n", q_f)

# Plots
plt.subplot(2, 1, 1)
plt.ylabel("err")
plt.semilogy(log_err, linestyle='-', color='blue')
plt.grid()
plt.subplot(2, 1, 2)
plt.ylabel("grad")
plt.xlabel("number of iterations")
plt.semilogy(log_grad, linestyle='-', color='blue')
plt.grid()
plt.show()

##################
# Polinomial Trajectory
##################
# Compute current pose at homing configuration
current_pos = T_wt[:3, 3]
R_current = T_wt[:3, :3]
rpy_current = math_utils.rot2eul(R_current)
current_roll = rpy_current[0]
current_pose = np.append(current_pos, current_roll)

pdes = np.array([1.0, 2.0, 1.0])
θdes = 30 * math.pi / 180 # 30 gradi
desired_pose = np.append(pdes, θdes)  # [x_des, y_des, z_des, roll_des]

duration = 3.0  # Trajectory duration in seconds
dt = 0.01       # Time step

# Generate coefficients for each component
coeffs = []
for i in range(4):  # x, y, z, roll
    c = coeffTraj(duration, current_pose[i], desired_pose[i])
    coeffs.append(c)

time_log = []
traj = []  # Each entry: [x(t), y(t), z(t), roll(t)]

t = 0.0
while t <= duration:
    point = []
    for i in range(4):
        a = coeffs[i]
        pos = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        point.append(pos)
    traj.append(point)
    time_log.append(t)
    t += dt

traj = np.array(traj)
plt.figure(figsize=(10, 8))
labels = ['X Position', 'Y Position', 'Z Position', 'Roll']
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(time_log, traj[:, i])
    plt.ylabel(labels[i])
    plt.grid(True)
plt.xlabel('Time (s)')
plt.suptitle('Task Space Trajectory')
plt.show()

ros_pub.deregister_node()
plt.show(block=True)








