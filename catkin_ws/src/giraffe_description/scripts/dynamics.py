#!/usr/bin/env python
#common stuff 
import pinocchio as pin
from pinocchio.utils import *
import numpy as np
from numpy import nan
import math
import time as tm

import os
from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.kin_dyn_utils import RNEA
from utils.kin_dyn_utils import getM
from utils.kin_dyn_utils import getg
from utils.kin_dyn_utils import getC

import conf as conf

# Init variables
zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
error = np.array([1, 1, 1, 1])

# Initialize logs with proper dimensions
def initialize_log(time, q, qd, qdd, q_des, qd_des, qdd_des):
    time_log = [time]
    q_log = [q.copy()]
    qd_log = [qd.copy()]
    qdd_log = [qdd.copy()]
    q_des_log = [q_des.copy()]
    qd_des_log = [qd_des.copy()]
    qdd_des_log = [qdd_des.copy()]
    tau_log = [np.zeros_like(q)]

    return time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log

# Main loop to simulate dynamics
def dyn_simulation(robot, time, ros_pub, q, qd, qdd, q_des, qd_des, qdd_des):
    time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log = initialize_log(time, q, qd, qdd, q_des, qd_des, qdd_des)
    while (not ros.is_shutdown()) and (time < conf.exp_dyn_duration):

        # initialize Pinocchio variables
        robot.computeAllTerms(q, qd)
        # vector of gravity acceleration
        g0 = np.array([0.0, 0.0, -9.81])
        # type of joints
        joint_types = np.array(['revolute', 'revolute', 'prismatic', 'revolute', 'revolute'])
        ##############################
        # Implement RNEA
        ##############################
        # compute RNEA with Pinocchio
        taup = pin.rnea(robot.model, robot.data, q, qd, qdd)
        print("RNEA: ", taup)
        print("-------------------------------")

        ######################################
        # Compute dynamic terms
        ######################################
        # Pinocchio
        gp = robot.gravity(q)
        print("Gravity: ", gp)
        print("-------------------------------")

        # compute joint space intertia matrix with built-in pinocchio rnea
        M  = np.zeros((5,5))
        for i in range(5):
            ei = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            ei[i] = 1
            taup = pin.rnea(robot.model, robot.data, q, np.array([0.0, 0.0, 0.0, 0.0, 0.0]) ,ei)
            M[:5,i] = taup - gp
        np.set_printoptions(suppress=True, precision=2)
        print("Inertia with Pinocchio RNEA: \n", M)
        print("-------------------------------")

        # Pinocchio bias terms (coriolis + gravity)
        hp = robot.nle(q, qd, False)
        print("Bias Term: ", hp)
        print("-------------------------------")

        #############################################
        # Add a damping term
        #############################################
        # viscous friction to stop the motion
        damping =  -0.1*qd

        end_stop_tau = np.zeros(5)
        # Total torque input
        total_tau = end_stop_tau + damping

        #############################################
        # Add end-stops
        #############################################
        jl_K = 10000
        jl_D = 10
        end_stop_tau = np.zeros(robot.nv)
        q_max = np.array([2*np.pi, 0.5, 6.5, 2*np.pi, 2*np.pi]) 
        q_min = np.array([-2*np.pi, -0.5, 0.0, -2*np.pi, -2*np.pi])

        # Calcolo della coppia di end-stop
        end_stop_tau =  (q > q_max) * (jl_K * (q_max - q) + jl_D * (-qd)) + \
                        (q < q_min) * (jl_K * (q_min - q) + jl_D * (-qd))

        # Total torque input
        total_tau = end_stop_tau + damping

        #############################################
        # Compute joint accelerations
        #############################################
        # compute accelerations (torques are zero!)
        # Pinocchio
        qdd = np.linalg.inv(M).dot(total_tau - hp)
        print("qdd computed by forward dynamics: ", qdd)

        qdd_p = pin.aba(robot.model, robot.data, q, qd, total_tau)
        print("qdd computed by ABA: ", qdd_p)

        #############################################
        # Simulate the forward dynamics
        #############################################
        # Forward Euler Integration
        qd = qd + qdd * conf.dt
        q = q + conf.dt * qd  + 0.5 * pow(conf.dt,2) * qdd

        # update time
        time = time + conf.dt

        # Log Data into a vector
        time_log.append(time)
        q_log.append(q.copy())
        qd_log.append(qd.copy())
        qdd_log.append(qdd.copy())
        q_des_log.append(q_des.copy())  # Constant desired
        qd_des_log.append(qd_des.copy())
        qdd_des_log.append(qdd_des.copy())
        tau_log.append(total_tau.copy())

        # M_log = np.dstack((M_log, M))
        # C_log = np.dstack((C_log, C))
        # g_log = np.dtack((g_log, g))
                    
        #publish joint variables
        ros_pub.publish(robot, q, qd)
        tm.sleep(conf.dt*conf.SLOW_FACTOR)

    # Convert to numpy arrays for plotting
    # Convert to numpy arrays for plotting
    time_log = np.array(time_log)
    q_log = np.array(q_log).T
    qd_log = np.array(qd_log).T
    qdd_log = np.array(qdd_log).T
    q_des_log = np.array(q_des_log).T
    qd_des_log = np.array(qd_des_log).T
    qdd_des_log = np.array(qdd_des_log).T
    tau_log = np.array(tau_log).T
                
    return time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log