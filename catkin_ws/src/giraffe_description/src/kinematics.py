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

import kin_conf as conf

#os.system("killall rosmaster rviz")
#instantiate graphic utils
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
kin = robotKinematics(robot, conf.frame_name)

# Init variables
zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
time = 0.0

# Init loggers
q_log = np.empty((0, 5))
qd_log = np.empty((0, 5))
qdd_log = np.empty((0, 5))
time_log = np.array([])

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

# q_ik, _, _, = kin.endeffectorInverseKinematicsLineSearch(p, conf.frame_name, q_i, verbose=True, use_error_as_termination_criteria=False, postural_task=False)
# print("Desired End effector \n", p)

# robot.computeAllTerms(q_ik, np.zeros(5))
# p_ik = robot.framePlacement(q_ik, robot.model.getFrameId(conf.frame_name)).translation
# task_diff = p_ik - p
# print("Point obtained with IK solution \n", p_ik)
# print("Error at the end-effector: \n", np.linalg.norm(task_diff))
# print("Final joint positions\n", q_ik)

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
tm.sleep(1.)
ros_pub.publish(robot, conf.q0)
tm.sleep(2.)
duration = 3.0
a_list = [coeffTraj(duration, conf.q0[i], q_f[i]) for i in range(5)]
q = np.array([0.0, -math.pi, 0.0, 0.0, -math.pi/2])
print("Diff between q and q_f:", np.linalg.norm(q - q_f))

while np.count_nonzero(q - q_f):
    # Polynomial trajectory
    for i in range(5):
        a = a_list[i]
        q[i]   = a[0] + a[1]*time + a[2]*time**2 + a[3]*time**3 + a[4]*time**4 + a[5]*time**5
        qd[i]  = a[1] + 2*a[2]*time + 3*a[3]*time**2 + 4*a[4]*time**3 + 5*a[5]*time**4
        qdd[i] = 2*a[2] + 6*a[3]*time + 12*a[4]*time**2 + 20*a[5]*time**3

    # update time
    time = time + conf.dt

    # Log Data into a vector
    time_log = np.append(time_log, time)
    q_log = np.vstack((q_log, q ))
    qd_log= np.vstack((qd_log, qd))
    qdd_log= np.vstack((qdd_log, qdd))

    #publish joint variables
    ros_pub.publish(robot, q, qd)
    ros_pub.add_marker(p)
    ros.sleep(conf.dt*conf.SLOW_FACTOR)

    # stops the while loop if  you prematurely hit CTRL+C
    if ros_pub.isShuttingDown():
        print ("Shutting Down")
        break

plotJoint('position', time_log, q_log.T)
ros_pub.deregister_node()
plt.show(block=True)








