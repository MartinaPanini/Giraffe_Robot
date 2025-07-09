# -*- coding: utf-8 -*-
"""
Created on May 4 2021

@author: ovillarreal
"""
from __future__ import print_function
import numpy as np
import os
import math
import pinocchio as pin
from pinocchio.utils import *
from utils.math_tools import Math
import time as tm 

def setRobotParameters():
    # Lunghezze dei link
    l1 = 0.0                      # base_link -> link1 (nessuna traslazione)
    l2 = 0.5                      # link1 -> link2 (lunghezza del braccio sferico)
    l3 = 6.5                      # link2 -> link3 (estensione massima del prismatico)
    l4 = 0.05                     # offset dal giunto prismatico a link4 (es. distanza da sfera a bar)
    l5 = 0.3                      # lunghezza della barra del microfono
    l6 = 0.03                     # offset dalla fine della barra al centro del box
    l7 = 0.2                      # lunghezza microfono (cylinder + sphere)

    lengths = np.array([l1, l2, l3, l4, l5, l6, l7])

    # Masse prese dall'URDF
    m0 = 10.0     # base_link
    m1 = 0.1      # link1
    m2 = 0.5      # link2
    m3 = 0.1      # link3
    m4 = 0.05     # link4 (mic bar)
    m5 = 0.01     # ee_link (microfono)

    link_masses = np.array([m0, m1, m2, m3, m4, m5])

    # Centri di massa (COM) in frame locali (da origin xyz nei tag <inertial>)
    com0 = np.array([0.0, 0.0, 0.0])                            # base_link
    com1 = np.array([0.0, 0.0, 0.0])                            # link1
    com2 = np.array([0.25, 0.0, 0.0])                           # link2 (lunghezza 0.5, origin a metà)
    com3 = np.array([0.0, 0.0, 0.0])                            # link3 (sfera, centro = origine)
    com4 = np.array([0.15, 0.0, 0.0])                           # link4 (barretta da 0.3m)
    com5 = np.array([0.0, 0.0, -0.1])                           # ee_link (centro della parte cilindrica)

    coms = np.array([com0, com1, com2, com3, com4, com5])

    # Tensori d'inerzia (w.r.t. il CoM in frame locali)
    I_0 = np.diag([1.0, 1.0, 1.0])
    I_1 = np.diag([0.001, 0.001, 0.001])
    I_2 = np.diag([0.01, 0.01, 0.01])
    I_3 = np.diag([0.001, 0.001, 0.001])
    I_4 = np.diag([0.0005, 0.0005, 0.0005])
    I_5 = np.diag([0.0001, 0.0001, 0.0001])

    inertia_tensors = np.array([I_0, I_1, I_2, I_3, I_4, I_5])

    return lengths, inertia_tensors, link_masses, coms

def directKinematics(q):

    def RotX(theta):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def RotY(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def RotZ(theta):return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def TransX(d): return np.array([[1, 0, 0, d],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    def TransZ(d): return np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, d],
                                    [0, 0, 0, 1]])
     
    def TransY(d): return np.array([[1, 0, 0, 0],
                                    [0, 1, 0, d],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        # define link lengths
    link_length, _, _, _ = setRobotParameters()
    l1, l2, l3, l4, l5, l6, l7 = link_length

    q1, q2, q3, q4, q5 = q

    # FIXED: Add base position from URDF (2.5, 6.0, 4.0)
    T_wb = np.eye(4)
    T_wb[0, 3] = 2.5
    T_wb[1, 3] = 6.0
    T_wb[2, 3] = 4.0  # ceiling height

    # Joint 1: base_link to link1
    T_b1 = RotY(math.pi) @ RotZ(q1)

    # Joint 2: link1 to link2
    T_12 = RotY(q2)

    # Joint 3: prismatic, origin at l2, motion along X
    T_23 = TransX(l2) @ TransX(q3)

    # Joint 4: rotation + offset 0.05 along X
    T_34 = RotY(q4) @ TransX(l4)

    # Joint 5: rotation + offset mic_arm_length (l4)
    T_4e = TransX(l5) @ RotY(q5)

    # End-effector tip
    #T_et = np.array([[1,  0, 0,  0],
    #                 [0,  1, 0,  0],
    #                 [0,  0, 1, -0.2],
    #                 [0,  0, 0,  1]])
    T_et = np.eye(4)

    # Compose global transforms
    T_w1 = T_wb @ T_b1
    T_w2 = T_w1 @ T_12
    T_w3 = T_w2 @ T_23
    T_w4 = T_w3 @ T_34
    T_we = T_w4 @ T_4e
    T_wt = T_we @ T_et

    return T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt

'''
    This function computes the Geometric Jacobian of the end-effector expressed in the base link frame 
'''
def computeEndEffectorJacobian(q):
    import numpy as np

    # Ottieni le trasformazioni globali da base a ogni giunto e all'end-effector
    T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt = directKinematics(q)

    # Positions in world frame
    p_wb = T_wb[:3, 3]
    p_w1 = T_w1[:3, 3]
    p_w2 = T_w2[:3, 3]
    p_w3 = T_w3[:3, 3]
    p_w4 = T_w4[:3, 3]
    p_we = T_we[:3, 3]
    p_wt = T_wt[:3, 3]  # microphone tip

    # Joint axes in world frame
    z0 = T_wb[:3, 2]  # base Z (fixed)
    z1 = T_w1[:3, 2]  # joint1 rotation axis
    z2 = T_w2[:3, 1]  # joint2 rotation axis
    z3 = T_w3[:3, 0]  # joint3 prismatic axis (X)
    z4 = T_w4[:3, 1]  # joint4 rotation axis
    z5 = T_we[:3, 1]  # joint5 rotation axis

    # Jacobian for linear velocity
    Jp0 = np.cross(z0, (p_wt - p_wb)).reshape(3, 1)
    Jp1 = np.cross(z1, (p_wt - p_w1)).reshape(3, 1)
    Jp2 = np.cross(z2, (p_wt - p_w2)).reshape(3, 1)
    Jp3 = z3.reshape(3, 1)  # prismatic joint
    Jp4 = np.cross(z4, (p_wt - p_w4)).reshape(3, 1)
    Jp5 = np.cross(z5, (p_wt - p_we)).reshape(3, 1)

    J_p = np.hstack((Jp1, Jp2, Jp3, Jp4, Jp5))

    # Jacobian for angular velocity
    Jo0 = z0.reshape(3, 1)
    Jo1 = z1.reshape(3, 1)
    Jo2 = z2.reshape(3, 1)
    Jo3 = np.zeros((3, 1))  # prismatic has no rotation
    Jo4 = z4.reshape(3, 1)
    Jo5 = z5.reshape(3, 1)

    J_o = np.hstack((Jo1, Jo2, Jo3, Jo4, Jo5))

    # Complete Jacobian (6x6)
    J = np.vstack((J_p, J_o))

    return J, z1, z2, z3, z4, z5



def geometric2analyticJacobian(J,T_0e):
    R_0e = T_0e[:3,:3]
    math_utils = Math()
    rpy_ee = math_utils.rot2eul(R_0e)
    roll = rpy_ee[0]
    pitch = rpy_ee[1]
    yaw = rpy_ee[2]

    # compute the mapping between euler rates and angular velocity
    T_w = np.array([[math.cos(yaw)*math.cos(pitch),  -math.sin(yaw), 0],
                    [math.sin(yaw)*math.cos(pitch),   math.cos(yaw), 0],
                    [             -math.sin(pitch),               0, 1]])

    T_a = np.array([np.vstack((np.hstack((np.identity(3), np.zeros((3,3)))),
                                          np.hstack((np.zeros((3,3)),np.linalg.inv(T_w)))))])


    J_a = np.dot(T_a, J)

    return J_a[0]

def numericalInverseKinematics(p_d, q0, line_search = False, wrap = False):
    math_utils = Math()

    # hyper-parameters
    epsilon = 1e-06 # Tolerance for stopping criterion
    lambda_ = 1e-08  # Regularization or damping factor (1e-08->0.01)
    max_iter = 200  # Maximum number of iterations
    # For line search only
    #gamma = 0.5
    beta = 0.5 # Step size reduction

    # initialization of variables
    iter = 0
    alpha = 1  # Step size
    log_grad = []
    log_err = []

    # Inverse kinematics with line search
    while True:
        # evaluate  the kinematics for q0
        J,_,_,_,_,_ = computeEndEffectorJacobian(q0)
        _, _, _, _,_,_, T_0e = directKinematics(q0)

        p_e = T_0e[:3,3]
        R = T_0e[:3,:3]
        rpy = math_utils.rot2eul(R)
        roll = rpy[0]
        p_e = np.append(p_e,roll)

        # error
        e_bar = p_e - p_d
        J_bar = geometric2analyticJacobian(J,T_0e)
        # take first 4 rows correspondent to our task
        J_bar = J_bar[:4,:]
        # evaluate the gradient
        grad = J_bar.T.dot(e_bar)

        log_grad.append(np.linalg.norm(grad))
        log_err.append(np.linalg.norm(e_bar))

        if np.linalg.norm(grad) < epsilon:
            print("IK Convergence achieved!, norm(grad) :", np.linalg.norm(grad) )
            print("Inverse kinematics solved in {} iterations".format(iter))     
            break
        if iter >= max_iter:                
            print("Warning: Max number of iterations reached, the iterative algorithm has not reached convergence to the desired precision. Error is:  ", np.linalg.norm(e_bar))
            break
        # Compute the error
        JtJ= np.dot(J_bar.T,J_bar) + np.identity(J_bar.shape[1])*lambda_
        JtJ_inv = np.linalg.inv(JtJ)
        P = JtJ_inv.dot(J_bar.T)
        dq = - P.dot(e_bar)

        if not line_search:
            q1 = q0 + dq * alpha
            q0 = q1
        else:
            print("Iter # :", iter)
            # line search loop
            while True:
                #update
                q1 = q0 + dq*alpha
                # evaluate  the kinematics for q1
                _, _, _, _, _, _, T_0e1 = directKinematics(q1)
                p_e1 = T_0e1[:3,3]
                R1 = T_0e1[:3,:3]
                rpy1 = math_utils.rot2eul(R1)
                roll1 = rpy1[0]
                p_e1 = np.append(p_e1,roll1)
                e_bar_new = p_e1 - p_d
                #print "e_bar1", np.linalg.norm(e_bar_new), "e_bar", np.linalg.norm(e_bar)

                error_reduction = np.linalg.norm(e_bar) - np.linalg.norm(e_bar_new)
                threshold = 0.0 # more restrictive gamma*alpha*np.linalg.norm(e_bar)

                if error_reduction < threshold:
                    alpha = beta*alpha
                    print (" line search: alpha: ", alpha)
                else:
                    q0 = q1
                    alpha = 1
                    break

        iter += 1
           

 
    # wrapping prevents from outputs outside the range -2pi, 2pi
    if wrap:
        for i in range(len(q0)):
            while q0[i] >= 2 * math.pi:
                q0[i] -= 2 * math.pi
            while q0[i] < -2 * math.pi:
                q0[i] += 2 * math.pi

    return q0, log_err, log_grad


def fifthOrderPolynomialTrajectory(tf,start_pos,end_pos, start_vel = 0, end_vel = 0, start_acc =0, end_acc = 0):

    # Matrix used to solve the linear system of equations for the polynomial trajectory
    polyMatrix = np.array([[1,  0,              0,               0,                  0,                0],
                           [0,  1,              0,               0,                  0,                0],
                           [0,  0,              2,               0,                  0,                0],
                           [1, tf,np.power(tf, 2), np.power(tf, 3),    np.power(tf, 4),  np.power(tf, 5)],
                           [0,  1,           2*tf,3*np.power(tf,2),   4*np.power(tf,3), 5*np.power(tf,4)],
                           [0,  0,              2,             6*tf, 12*np.power(tf,2),20*np.power(tf,3)]])
    
    polyVector = np.array([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])
    matrix_inv = np.linalg.inv(polyMatrix)
    polyCoeff = matrix_inv.dot(polyVector)

    return polyCoeff
    
def RNEA(g0,q,qd,qdd, Fee = np.zeros(3), Mee = np.zeros(3), joint_types = ['revolute', 'revolute','revolute','revolute']):

    # setting values of inertia tensors w.r.t. to their CoMs from urdf and link masses
    _, tensors, m, coms = setRobotParameters()

    # get inertia tensors about the CoM expressed in the respective link frame
    _0_I_0 = tensors[0]
    _1_I_1 = tensors[1]
    _2_I_2 = tensors[2]
    _3_I_3 = tensors[3]
    _4_I_4 = tensors[4]
    
    # get positions of the link CoM expressed in the respective link frame
    _0_com_0 = coms[0]
    _1_com_1 = coms[1]
    _2_com_2 = coms[2]
    _3_com_3 = coms[3]
    _4_com_4 = coms[4]


    # number of joints
    n = len(q)
    
    #pre-pend a fake joint for base link
    q_link = np.insert(q, 0, 0.0, axis=0)
    qd_link = np.insert(qd, 0, 0.0, axis=0)
    qdd_link = np.insert(qdd, 0, 0.0, axis=0)
        
    # initialation of variables
    zeroV = np.zeros(3)
    omega = np.array([zeroV, zeroV, zeroV, zeroV, zeroV])
    v = np.array([zeroV, zeroV, zeroV, zeroV, zeroV])
    omega_dot = np.array([zeroV, zeroV, zeroV, zeroV,zeroV])
    a = np.array([zeroV, zeroV, zeroV, zeroV, zeroV])
    vc = np.array([zeroV, zeroV, zeroV, zeroV, zeroV])
    ac = np.array([zeroV, zeroV, zeroV, zeroV,zeroV])

    # these arrays are 1 element longer than the others because in the back recursion we consider also the forces/moments coming from the ee
    F = np.array([zeroV, zeroV, zeroV, zeroV, zeroV, Fee])
    M = np.array([zeroV, zeroV, zeroV, zeroV, zeroV, Mee])

    effort = np.array([0.0, 0.0, 0.0, 0.0])

    # obtaining joint axes vectors required in the computation of the velocities and accelerations (expressed in the world frame)
    _,z1,z2,z3,z4 = computeEndEffectorJacobian(q)

    z = np.array([np.zeros(3), z1,z2,z3,z4])

    # global homogeneous transformation matrices
    T_01, T_02, T_03, T_04, T_0e = directKinematics(q)

    # link positions w.r.t. the world frame
    p_00 = np.array([0.0,0.0,0.0])
    p_01 = T_01[:3,3]
    p_02 = T_02[:3,3]
    p_03 = T_03[:3,3]
    p_04 = T_04[:3,3]
    p_0e = T_0e[:3,3]

    # array used in the recursion (this array is 1 element longer than the others because in the back recursion we consider also the position of the ee)
    p = np.array([p_00, p_01, p_02, p_03, p_04, p_0e])

    # rotation matrices w.r.t. to the world of each link
    R_00 = np.eye(3)    
    R_01 = T_01[:3,:3]
    R_02 = T_02[:3,:3]
    R_03 = T_03[:3,:3]
    R_04 = T_04[:3,:3]

    # positions of the CoMs w.r.t. to the world frame
    pc_0 = p_00 + _0_com_0
    pc_1 = p_01 + np.dot(R_01, _1_com_1)
    pc_2 = p_02 + np.dot(R_02, _2_com_2)
    pc_3 = p_03 + np.dot(R_03, _3_com_3)
    pc_4 = p_04 + np.dot(R_04, _4_com_4)

    # array used in the recursion
    pc = np.array([pc_0, pc_1, pc_2, pc_3, pc_4])

    # expressing tensors of inertia of the links (about the com) in the world frame (time consuming)
    I_0 = np.dot(np.dot(R_00,_0_I_0),R_00.T)
    I_1 = np.dot(np.dot(R_01,_1_I_1),R_01.T)
    I_2 = np.dot(np.dot(R_02,_2_I_2),R_02.T)
    I_3 = np.dot(np.dot(R_03,_3_I_3),R_03.T)
    I_4 = np.dot(np.dot(R_04,_4_I_4),R_04.T)

    # array used in the recursion
    I = np.array([I_0, I_1, I_2, I_3, I_4])

    # forward pass: compute accelerations from link 0 to  link 4, range(n+1) = (0, 1, 2, 3, 4)
    for i in range(n+1):
        joint_idx = i-1
        if i == 0: # we start from base link 0
            p_ = p[0]
            #base frame is still (not true for a legged robot!)
            omega[0] = zeroV
            v[0] = zeroV
            omega_dot[0] = zeroV
            a[0] = -g0 # if we consider gravity as  acceleration (need to move to right hand side of the Newton equation) we can remove it from all the Netwon equations
        else:
            if joint_types[joint_idx] == 'prismatic':  # prismatic joint
                p_ = p[i] - p[i - 1]
                omega[i] = omega[i - 1]
                omega_dot[i] = omega_dot[i - 1]
                v[i] = v[i - 1] + qd_link[i] * z[i] + np.cross(omega[i], p_)
                a[i] = a[i - 1] + qdd_link[i] * z[i] + 2 * qd_link[i] * np.cross(omega[i], z[i]) + np.cross(omega_dot[i], p_) + np.cross(omega[i], np.cross(omega[i], p_))
            elif joint_types[joint_idx] == 'revolute':
                p_ = p[i] - p[i - 1]
                omega[i] = omega[i - 1] + qd_link[i] * z[i]
                omega_dot[i] = omega_dot[i - 1] + qdd_link[i] * z[i] + qd_link[i] * np.cross(omega[i - 1], z[i])
                v[i] = v[i - 1] + np.cross(omega[i - 1], p_)
                a[i] = a[i - 1] + np.cross(omega_dot[i - 1], p_) + np.cross(omega[i - 1], np.cross(omega[i - 1], p_))
            else:
                print("wrong joint type")
        pc_ = pc[i] - p[i] # p_i,c
        
        #compute com quantities
        vc[i] = v[i] + np.cross(omega[i],p_)
        ac[i] = a[i] + np.cross(omega_dot[i],pc_) + np.cross(omega[i],np.cross(omega[i],pc_))

    
    # backward pass: compute forces and moments from wrist link (4) to base link (0)
    for i in range(n,-1,-1):   
        # lever arms wrt to other link frames
        pc_ = p[i] - pc[i]
        pc_1 = p[i+1] - pc[i] 
        
        F[i] = F[i+1] + m[i]*(ac[i])
        
        M[i] = M[i+1] - \
               np.cross(pc_,F[i]) + \
               np.cross(pc_1,F[i+1]) + \
               np.dot(I[i],omega_dot[i]) + \
               np.cross(omega[i],np.dot(I[i],omega[i]))  

    # compute torque for all joints (revolute) by projection
    for joint_idx in range(n):
        if joint_types[joint_idx] == 'prismatic':
            effort[joint_idx] = np.dot(z[joint_idx + 1], F[joint_idx + 1])
        elif joint_types[joint_idx] == 'revolute':
            effort[joint_idx] = np.dot(z[joint_idx + 1], M[joint_idx + 1])
        else:
            print("wrong joint type")
    return effort

# computation of gravity terms
def getg(q,robot, joint_types = ['revolute', 'revolute','revolute','revolute']):
    qd = np.array([0.0, 0.0, 0.0, 0.0])
    qdd = np.array([0.0, 0.0, 0.0, 0.0])
    # Pinocchio
    # g = pin.rnea(robot.model, robot.data, q,qd ,qdd)
    g = RNEA(np.array([0.0, 0.0, -9.81]),q,qd,qdd, joint_types=joint_types)
    return g


# computation of generalized mass matrix
def getM(q,robot, joint_types = ['revolute', 'revolute','revolute','revolute']):
    n = len(q)
    M = np.zeros((n,n))
    for i in range(n):
        ei = np.array([0.0, 0.0, 0.0, 0.0])
        ei[i] = 1
        # Pinocchio
        #g = getg(q,robot)
        # tau_p = pin.rnea(robot.model, robot.data, q, np.array([0,0,0,0]),ei) -g      
        tau = RNEA(np.array([0.0, 0.0, 0.0]), q, np.array([0.0, 0.0, 0.0, 0.0]),ei, joint_types=joint_types)
        # fill in the column of the inertia matrix
        M[:4,i] = tau        
        
    return M

def getC(q,qd,robot, joint_types = ['revolute', 'revolute','revolute','revolute']):
    qdd = np.array([0.0, 0.0, 0.0, 0.0])
    # Pinocchio
    # g = getg(q,robot)
    # C = pin.rnea(robot.model, robot.data,q,qd,qdd) - g    
    C = RNEA(np.array([0.0, 0.0, 0.0]), q, qd, qdd, joint_types=joint_types)
    return C      

    



