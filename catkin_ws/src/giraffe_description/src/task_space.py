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

# Initialize robot model
os.system("killall rosmaster rviz &> /dev/null")
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
model = robot.model
data = robot.data
math_utils = Math()

# Initialize variables
q = conf.qhome.copy()
qd = np.zeros(robot.nv)
qdd = np.zeros(robot.nv)
time = 0.0
log_counter = 0

# Get frame ID
assert(model.existFrame(conf.frame_name))
frame_id = model.getFrameId(conf.frame_name)

# Precompute initial end-effector position (FIX 1: Correct trajectory reference)
pin.framesForwardKinematics(model, data, conf.qhome)
p0 = data.oMf[frame_id].translation.copy()

# Logging setup
buffer_size = int(math.ceil(conf.exp_duration / conf.dt))
q_log = np.zeros((robot.nq, buffer_size))
p_log = np.zeros((3, buffer_size))
p_des_log = np.zeros((3, buffer_size))
rpy_log = np.zeros((3, buffer_size)) 
rpy_des_log = np.zeros((3, buffer_size))
time_log = np.zeros(buffer_size)

# Desired orientation: -30 degree pitch (rotation around Y-axis) to point downwards.
pitch_angle_rad = np.radians(-30.0)
w_R_des = pin.rpy.rpyToMatrix(0, pitch_angle_rad, 0)

# Generate trajectory with fixed start point (FIX 1)
def generate_trajectory(t):
    if t > conf.traj_duration:
        return conf.pdes, np.zeros(3), np.zeros(3)
    
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    
    for i in range(3):
        c = coeffTraj(conf.traj_duration, p0[i], conf.pdes[i])
        pos[i] = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
        vel[i] = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
        acc[i] = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
    
    return pos, vel, acc

# Main control loop
print("Starting task-space control...")
while time < conf.exp_duration:
    if log_counter >= buffer_size:
        break
    
    # Get reference trajectory
    p_des, v_des, a_des = generate_trajectory(time)
    
    # Compute current state
    pin.forwardKinematics(model, data, q, qd)
    pin.updateFramePlacement(model, data, frame_id)
   
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
    p = data.oMf[frame_id].translation
    w_R_e = data.oMf[frame_id].rotation
    twist = J @ qd
    v = twist[:3]
    omega = twist[3:]
    
    # Task-space error
    p_error = p_des - p
    v_error = v_des - v

    e_R_des = w_R_e.T @ w_R_des
    error_o_local = pin.log3(e_R_des)
    w_error_o = w_R_e @ error_o_local
    omega_error = -omega
    
    # Desired task acceleration (FIX 2: Ensure proper gains)
    a_task = a_des + conf.Kd_task @ v_error + conf.Kp_task @ p_error
    alpha_task = conf.Kp_ori @ w_error_o + conf.Kd_ori @ omega_error
    acc_des = np.hstack([a_task, alpha_task])
    
    # Compute Jacobian and its time derivative
    # Full 6D Inverse Dynamics
    Jdot = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    Jdotqd = Jdot @ qd
    
    # Compute dynamics
    M = pin.crba(model, data, q)
    h = pin.nonLinearEffects(model, data, q, qd)
    
    # Task-space inverse dynamics
    Minv = np.linalg.inv(M)
    Lambda = np.linalg.inv(J @ Minv @ J.T + 1e-6*np.eye(6))
    Jbar = Minv @ J.T @ Lambda
    
    # FIX 3: Proper null-space projection
    q_postural = conf.Kp_postural * (conf.q0 - q) - conf.Kd_postural * qd
    N = np.eye(robot.nv) - J.T @ np.linalg.pinv(J.T, 1e-4)  # Proper null-space projection
    
    # Combined desired acceleration
    qdd_des = Jbar @ (acc_des - Jdotqd) + N @ q_postural  # FIX 4: Include Minv
    
    # Torque command
    tau = M @ qdd_des + h
    
    # Forward dynamics simulation (FIX 5: Use proper integration)
    qdd = Minv @ (tau - h)
    qd_next = qd + qdd * conf.dt
    q_next = pin.integrate(model, q, (qd + qd_next) * 0.5 * conf.dt)  # Trapezoidal integration
    
    # Update state
    q = q_next
    qd = qd_next
    
    # Log data
    time_log[log_counter] = time
    q_log[:, log_counter] = q
    p_log[:, log_counter] = p
    p_des_log[:, log_counter] = p_des
    rpy_log[:, log_counter] = pin.rpy.matrixToRpy(w_R_e)
    rpy_des_log[:, log_counter] = pin.rpy.matrixToRpy(w_R_des)
    log_counter += 1
    
    # Update time
    time += conf.dt
    
    # Visualize
    ros_pub.publish(robot, q, qd, tau)
    tm.sleep(conf.dt * conf.SLOW_FACTOR)
    
    if ros_pub.isShuttingDown():
        break

# Calcola e stampa la posa finale dell'end effector
pin.framesForwardKinematics(model, data, q)
pin.updateFramePlacement(model, data, frame_id)
final_placement = data.oMf[frame_id]
final_position = final_placement.translation
final_orientation_rpy = pin.rpy.matrixToRpy(final_placement.rotation)
desired_orientation_rpy = pin.rpy.matrixToRpy(w_R_des)

print("\nFinal End Effector Position (m):", final_position)
print("Final End Effector Orientation (RPY - deg):", np.degrees(final_orientation_rpy))
print("Desired End Effector Orientation (RPY - deg):", np.degrees(desired_orientation_rpy))

# Plotting
plt.figure(figsize=(12, 12))
plt.suptitle("Task-Space Control Performance")

# Position tracking
plt.subplot(3, 1, 1)
labels = ['X', 'Y', 'Z']
for i in range(3):
    plt.plot(time_log[:log_counter], p_log[i, :log_counter], label=f'Actual {labels[i]}')
    plt.plot(time_log[:log_counter], p_des_log[i, :log_counter], '--', label=f'Desired {labels[i]}')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)

# Orientation tracking (RPY)
plt.subplot(3, 1, 2)
labels = ['Roll', 'Pitch', 'Yaw']
for i in range(3):
    plt.plot(time_log[:log_counter], np.degrees(rpy_log[i, :log_counter]), label=f'Actual {labels[i]}')
    plt.plot(time_log[:log_counter], np.degrees(rpy_des_log[i, :log_counter]), '--', label=f'Desired {labels[i]}')
plt.ylabel('Orientation [deg]')
plt.legend()
plt.grid(True)

# Configuration tracking
plt.subplot(3, 1, 3)
for i in range(robot.nq):
    plt.plot(time_log[:log_counter], q_log[i, :log_counter], label=f'Joint {i+1}')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angles [rad]')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

input("Press Enter to terminate...")

print("Task-space control completed.")
ros_pub.deregister_node()