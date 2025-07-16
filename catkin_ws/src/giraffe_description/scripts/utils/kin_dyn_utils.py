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
    