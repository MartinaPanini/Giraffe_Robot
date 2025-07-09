#!/usr/bin/env python
# coding=utf-8 
from __future__ import print_function

import pinocchio as pin
import numpy as np
import math
import time as tm
import matplotlib.pyplot as plt
import os

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.math_tools import Math
from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj
import task_conf as conf

# Requirement 5: Task-space inverse dynamics control
# Requirement 8: Reach pdes from qhome

# Initialize robot model
os.system("killall rosmaster rviz &> /dev/null")
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
model = robot.model
data = robot.data
math_utils = Math()

# Initialize variables
q = conf.qhome.copy()          # Start from homing config
qd = np.zeros(robot.nv)
qdd = np.zeros(robot.nv)
time = 0.0
log_counter = 0

# Get frame ID
assert(model.existFrame(conf.frame_name))
frame_id = model.getFrameId(conf.frame_name)

# Logging setup
buffer_size = int(conf.exp_duration/conf.dt)
q_log = np.zeros((robot.nq, buffer_size))
p_log = np.zeros((3, buffer_size))
p_des_log = np.zeros((3, buffer_size))
time_log = np.zeros(buffer_size)

# Requirement 8: Generate trajectory to pdes
def generate_trajectory(t):
    """5th-order polynomial trajectory from start to pdes"""
    if t > conf.traj_duration:
        return conf.pdes, np.zeros(3), np.zeros(3)
    
    # Compute trajectory for each component
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    
    # Get current position at t=0
    pin.framesForwardKinematics(model, data, q)
    p_start = data.oMf[frame_id].translation
    
    for i in range(3):
        # Coefficients for 5th-order polynomial
        c = coeffTraj(
            conf.traj_duration, 
            p_start[i], 
            conf.pdes[i]
        )
        # Position
        pos[i] = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
        # Velocity
        vel[i] = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
        # Acceleration
        acc[i] = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
    
    return pos, vel, acc

# Main control loop
print("Starting task-space control...")
while time < conf.exp_duration:
    # Get reference trajectory (Requirement 8)
    p_des, v_des, a_des = generate_trajectory(time)
    
    # Compute current end-effector position and velocity
    pin.forwardKinematics(model, data, q, qd)
    pin.updateFramePlacement(model, data, frame_id)
    p = data.oMf[frame_id].translation
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3,:]
    v = J @ qd
    
    # Requirement 5: Task-space control law
    # Task-space error
    p_error = p_des - p
    v_error = v_des - v
    
    # Desired task acceleration (PD controller)
    a_task = a_des + conf.Kd_task @ v_error + conf.Kp_task @ p_error
    
    # Compute Jacobian and its time derivative
    Jdot = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3,:]
    
    # Compute joint-space inertia matrix and bias
    M = pin.crba(model, data, q)
    h = pin.nonLinearEffects(model, data, q, qd)
    
    # Requirement 5: Task-space inverse dynamics
    # Compute dynamically consistent inverse
    Minv = np.linalg.inv(M)
    Lambda = np.linalg.inv(J @ Minv @ J.T + 1e-6*np.eye(3))  # Regularized
    Jbar = Minv @ J.T @ Lambda
    
    # Requirement 7: Postural task in null-space
    q_postural = conf.Kp_postural * (conf.q0 - q) - conf.Kd_postural * qd
    N = np.eye(robot.nv) - Jbar @ J  # Null-space projector
    
    # Combined desired acceleration
    qdd_des = Jbar @ (a_task - Jdot @ qd) + N @ q_postural
    
    # Torque command (computed torque)
    tau = M @ qdd_des + h
    
    # Forward dynamics simulation
    qdd = np.linalg.inv(M) @ (tau - h)
    qd += qdd * conf.dt
    q = pin.integrate(model, q, qd * conf.dt)
    
    # Log data
    time_log[log_counter] = time
    q_log[:, log_counter] = q
    p_log[:, log_counter] = p
    p_des_log[:, log_counter] = p_des
    log_counter += 1
    
    # Update time
    time += conf.dt
    
    # Visualize
    ros_pub.publish(robot, q, qd, tau)
    tm.sleep(conf.dt * conf.SLOW_FACTOR)
    
    if ros_pub.isShuttingDown():
        break

# Plot results (Requirement 8)
plt.figure(figsize=(12, 8))
plt.suptitle("Task-Space Control Performance")

# Position tracking
plt.subplot(2, 1, 1)
labels = ['X', 'Y', 'Z']
for i in range(3):
    plt.plot(time_log, p_log[i, :], label=f'Actual {labels[i]}')
    plt.plot(time_log, p_des_log[i, :], '--', label=f'Desired {labels[i]}')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)

# Configuration tracking
plt.subplot(2, 1, 2)
for i in range(robot.nq):
    plt.plot(time_log, q_log[i, :], label=f'Joint {i+1}')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angles [rad]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Task-space control completed.")
ros_pub.deregister_node()