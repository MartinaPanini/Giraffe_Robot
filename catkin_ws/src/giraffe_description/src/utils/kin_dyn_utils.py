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
    """
    Calcola la cinematica diretta basandosi sulla struttura esatta del file URDF.
    Ogni matrice di trasformazione T_i(i+1) mappa dal frame i al frame i+1.
    """
    # Helper per le trasformazioni omogenee
    def RotX(th): return pin.SE3(pin.rpy.rpyToMatrix(th, 0, 0), np.zeros(3)).homogeneous
    def RotY(th): return pin.SE3(pin.rpy.rpyToMatrix(0, th, 0), np.zeros(3)).homogeneous
    def RotZ(th): return pin.SE3(pin.rpy.rpyToMatrix(0, 0, th), np.zeros(3)).homogeneous
    def Trans(x,y,z): return pin.SE3(np.eye(3), np.array([x,y,z])).homogeneous

    # Parametri presi dal file URDF
    ceiling_height = 4.0
    arm_segment_length = 0.5
    mic_arm_length = 0.3
    
    # Valori dei giunti
    q1, q2, q3, q4, q5 = q

    # --- Catena di Trasformazioni come da URDF ---

    # 1. World -> base_link (Fisso)
    T_w_b = Trans(2.5, 6.0, ceiling_height)

    # 2. base_link -> link1 (joint1: offset in z, rotazione in z)
    T_b_1 = Trans(0, 0, 0.025) @ RotZ(q1)

    # 3. link1 -> link2 (joint2: rotazione in y)
    T_1_2 = RotY(q2)

    # 4. link2 -> link3 (joint3: offset in x, traslazione prismatica in x)
    T_2_3 = Trans(arm_segment_length, 0, 0) @ Trans(q3, 0, 0)

    # 5. link3 -> link4 (joint4: rotazione in y)
    T_3_4 = RotY(q4)

    # 6. link4 -> ee_link (joint5: offset in x, rotazione in y)
    T_4_e = Trans(mic_arm_length, 0, 0) @ RotY(q5)

    # Matrici di Trasformazione Composte (dal mondo a ogni frame)
    T_w_1 = T_w_b @ T_b_1
    T_w_2 = T_w_1 @ T_1_2
    T_w_3 = T_w_2 @ T_2_3
    T_w_4 = T_w_3 @ T_3_4
    T_w_e = T_w_4 @ T_4_e # Questa è la posa finale dell'end-effector (ee_link)

    # T_wt è la stessa di T_w_e dato che ee_link è il nostro frame finale
    T_wt = T_w_e 

    # Ritorna tutte le trasformazioni intermedie per il calcolo del Jacobiano
    return T_w_b, T_w_1, T_w_2, T_w_3, T_w_4, T_w_e, T_wt

'''
    This function computes the Geometric Jacobian of the end-effector expressed in the base link frame 
'''
def computeEndEffectorJacobian(q):
    """
    Calcola il Jacobiano Geometrico dell'end-effector.
    CORREZIONE: Usa le posizioni corrette delle origini dei giunti.
    """
    T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt = directKinematics(q)

    # Posizione finale dell'end-effector
    p_we = T_wt[:3, 3]

    # --- Posizioni delle ORIGINI DEI GIUNTI (non dei link) ---
    # Per ottenere l'origine di un giunto, prendiamo la trasformazione del link padre
    # e la componiamo con l'offset definito nel tag <origin> del giunto.
    
    # Origine di joint1 (fissato rispetto a base_link)
    p_j1 = (T_wb @ pin.SE3(np.eye(3), np.array([0, 0, 0.025])).homogeneous)[:3, 3]
    # Origine di joint2 (coincide con l'origine di link1)
    p_j2 = T_w1[:3, 3]
    # Origine di joint3 (offset da link2)
    p_j3 = (T_w2 @ pin.SE3(np.eye(3), np.array([0.5, 0, 0])).homogeneous)[:3, 3]
    # Origine di joint4 (coincide con l'origine di link3)
    p_j4 = T_w3[:3, 3]
    # Origine di joint5 (offset da link4)
    p_j5 = (T_w4 @ pin.SE3(np.eye(3), np.array([0.3, 0, 0])).homogeneous)[:3, 3]

    # --- Assi dei Giunti ---
    z1 = T_wb[:3, 2]   # Asse Z di base_link
    z2 = T_w1[:3, 1]   # Asse Y di link1
    z3 = T_w2[:3, 0]   # Asse X di link2 (Prismatico)
    z4 = T_w3[:3, 1]   # Asse Y di link3
    z5 = T_w4[:3, 1]   # Asse Y di link4

    # --- Colonne del Jacobiano ---
    Jp1 = np.cross(z1, p_we - p_j1); Jo1 = z1
    Jp2 = np.cross(z2, p_we - p_j2); Jo2 = z2
    Jp3 = z3;                        Jo3 = np.zeros(3) # Prismatico
    Jp4 = np.cross(z4, p_we - p_j4); Jo4 = z4
    Jp5 = np.cross(z5, p_we - p_j5); Jo5 = z5

    J_p = np.vstack([Jp1, Jp2, Jp3, Jp4, Jp5]).T
    J_o = np.vstack([Jo1, Jo2, Jo3, Jo4, Jo5]).T
    J = np.vstack((J_p, J_o))

    return J, z1, z2, z3, z4, z5

def geometric2analyticJacobian(J,T_0e):
    R_0e = T_0e[:3,:3]
    math_utils = Math()
    # ATTENZIONE: rot2eul potrebbe usare una convenzione diversa da Pinocchio.
    # Per coerenza, è sempre meglio usare le funzioni di Pinocchio se possibile.
    try:
        rpy_ee = pin.rpy.matrixToRpy(R_0e)
    except: # Fallback se la matrice non è di rotazione pura
        rpy_ee = np.zeros(3)

    roll, pitch, yaw = rpy_ee[0], rpy_ee[1], rpy_ee[2]

    # Matrice di mappatura T_w
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    # Questa è la mappatura per angoli ZYX (Yaw, Pitch, Roll)
    # Assicurati che questa sia la convenzione che desideri
    T_w_inv = np.array([
        [cy/cp, sy/cp, 0],
        [-sy,      cy, 0],
        [cy*sp/cp, sy*sp/cp, 1]
    ])
    
    T_a = np.block([
        [np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), T_w_inv]
    ])

    J_a = T_a @ J
    return J_a

def numericalInverseKinematics(p_d, q0, line_search=False, wrap=False, q_postural=None, kp_postural=0.5):
    """
    Computes the inverse kinematics for a desired 4D pose (x, y, z, pitch).

    This final version includes a robust backtracking line search to ensure convergence.
    """
    epsilon = 1e-6
    lambda_ = 1e-8
    max_iter = 2000
    beta = 0.5
    iter = 0

    log_grad = []
    log_err = []

    while True:
        J_geom, _, _, _, _, _ = computeEndEffectorJacobian(q0)
        T_0e = directKinematics(q0)[5]

        p_e = T_0e[:3, 3]
        try:
            rpy = pin.rpy.matrixToRpy(T_0e[:3, 3])
        except:
            rpy = np.zeros(3)

        current_pose_4d = np.array([p_e[0], p_e[1], p_e[2], rpy[1]])

        e_pos = current_pose_4d[:3] - p_d[:3]
        e_ori = current_pose_4d[3] - p_d[3]
        e_ori_wrapped = (e_ori + np.pi) % (2 * np.pi) - np.pi
        e_bar = np.hstack([e_pos, e_ori_wrapped])

        J_a = geometric2analyticJacobian(J_geom, T_0e)
        J_bar = np.vstack([J_a[0:3, :], J_a[4, :]])

        grad = J_bar.T @ e_bar

        log_grad.append(np.linalg.norm(grad))
        log_err.append(np.linalg.norm(e_bar))

        if np.linalg.norm(e_bar) < epsilon:
            print("IK Convergence achieved!, norm(error) :", np.linalg.norm(e_bar))
            print("Inverse kinematics solved in {} iterations".format(iter))
            break
        if iter >= max_iter:
            print("Warning: Max number of iterations reached. Error is: ", np.linalg.norm(e_bar))
            break

        JtJ_inv = np.linalg.inv(J_bar.T @ J_bar + lambda_ * np.identity(J_bar.shape[1]))
        dq = - JtJ_inv @ grad

        if q_postural is not None:
            J_bar_pinv = np.linalg.pinv(J_bar)
            N = np.eye(J_bar.shape[1]) - J_bar_pinv @ J_bar
            e_postural = q_postural - q0
            dq_secondary = kp_postural * e_postural
            dq += N @ dq_secondary

        # --- FIX: Robust Backtracking Line Search ---
        alpha = 1.0
        initial_error_norm = np.linalg.norm(e_bar)

        if line_search:
            while alpha > 1e-5:
                q_new = q0 + alpha * dq
                T_0e1 = directKinematics(q_new)[5]
                p_e1 = T_0e1[:3, 3]
                try:
                    rpy1 = pin.rpy.matrixToRpy(T_0e1[:3, 3])
                except:
                    rpy1 = np.zeros(3)

                current_pose_4d_new = np.array([p_e1[0], p_e1[1], p_e1[2], rpy1[1]])
                e_pos_new = current_pose_4d_new[:3] - p_d[:3]
                e_ori_new = current_pose_4d_new[3] - p_d[3]
                e_ori_wrapped_new = (e_ori_new + np.pi) % (2 * np.pi) - np.pi
                e_bar_new = np.hstack([e_pos_new, e_ori_wrapped_new])

                if np.linalg.norm(e_bar_new) < initial_error_norm:
                    q0 = q_new
                    break
                alpha *= beta
        else:
            q0 += dq
        # --- END FIX ---

        iter += 1

    if wrap:
        for i in range(len(q0)):
            q0[i] = (q0[i] + np.pi) % (2 * np.pi) - np.pi

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

    



