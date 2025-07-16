#!/usr/bin/env python
from __future__ import print_function

import numpy as np 
import matplotlib.pyplot as plt 
import pinocchio as pin 

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.in_kin_pinocchio import robotKinematics
from utils.math_tools import Math

from kinematics import *
from task_space import simulation 
from dynamics import dyn_simulation 

import conf as conf

ros_pub = None
robot = getRobotModel("giraffe", generate_urdf=True)
data = robot.data
model = robot.model
kin = robotKinematics(robot, conf.frame_name)

# Init common variables
q = conf.q0.copy()
qd = conf.qd0.copy()
qdd = conf.qdd0.copy()

q_des = conf.q0
qd_des = conf.qd0
qdd_des = conf.qdd0

math_utils = Math()

time = 0.0
zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# get the ID corresponding to the frame we want to control
assert(robot.model.existFrame(conf.frame_name))
frame_id = robot.model.getFrameId(conf.frame_name)

print("\n--- Choose an option ---")
print("1: Test Direct and Forward Kinematics")
print("2: Simulation with RNEA")
print("3: Simulation in Task Space")
print("4: Robot Visualization (RViz Only)")
choice = input("Choose the number corresponding to the test/simulation desired: ")

# Variables for dynamics plots
time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log = \
    None, None, None, None, None, None, None, None

if choice == '1':
    print("\nTest Forward and Differential kinematics")
    #####################################################################
    # Test Direct Kinematics
    #####################################################################
    dk_test(q, qd, robot, frame_id)
    jacobian_test(q, frame_id, robot)

elif choice == '2':
    print("\nSimulation with RNEA")
    #####################################################################
    # Test Dynamics
    #####################################################################
    ros_pub = RosPub("giraffe_robot")
    time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log = \
        dyn_simulation(robot, time, ros_pub, q, qd, qdd, q_des, qd_des, qdd_des)

elif choice == '3':
    print("\nSimulation of task space")
    #####################################################################
    # Test Simulation (Task Space)
    #####################################################################
    ros_pub = RosPub("giraffe_robot")
    
    q_sim = conf.qhome.copy()
    qd_sim = np.zeros(robot.nv)
    pitch_des_final = np.radians(conf.pitch_des_deg)

    q_final, qd_final = simulation(ros_pub, robot, model, data, pitch_des_final, q_sim, qd_sim)

    # Update forward kinematiccs 
    pin.forwardKinematics(model, data, q_final, qd_final)
    pin.updateFramePlacement(model, data, frame_id)

    # Print final pose
    final_pos = data.oMf[frame_id].translation
    final_orient_rpy = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
    print("\nFinal Position of the End Effector (m):", final_pos)
    print("Final Orientation of the End-Effector (RPY - deg):", np.degrees(final_orient_rpy))
    print("Pitch final (deg):", np.degrees(final_orient_rpy[1]))
    print("Pitch desired (deg):", np.degrees(pitch_des_final))

elif choice == '4': 
    print("\nRobot visualization in RViz")
    #####################################################################
    # Visualizzazione Robot (RViz Only)
    #####################################################################
    
    ros_pub = RosPub("giraffe_robot")
    ros_pub.publish(robot, conf.q0, np.zeros(robot.nv)) 
    print("RViz start. Press Ctrl+C to stop.")
    
    while ros_pub is not None and not ros_pub.isShuttingDown():
        try:
            ros.sleep(1.0) 
        except ros.ROSInterruptException:
            break
    print("RViZ visualization ended.")

else:
    print("CHOICE NOT VALID.")


if ros_pub is not None:
    ros_pub.deregister_node()

# Plot only for dynamics simulation
if time_log is not None and len(time_log) > 0:
    print("\nPlots generation...")
    plt.figure(1)
    plotJoint('position', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(2)
    plotJoint('velocity', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(3)
    plotJoint('acceleration', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(4)
    plotJoint('torque', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    
    plt.show()

    input("Press Enter to continue.")

input("Program ended.")