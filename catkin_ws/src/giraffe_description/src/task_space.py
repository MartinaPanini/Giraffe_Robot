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

# Inizializzazione del modello del robot
os.system("killall rosmaster rviz &> /dev/null")
ros_pub = RosPub("giraffe_robot")
robot = getRobotModel("giraffe", generate_urdf=True)
model = robot.model
data = robot.data
math_utils = Math()

# Inizializzazione delle variabili
q = conf.qhome.copy()
qd = np.zeros(robot.nv)
qdd = np.zeros(robot.nv)
time = 0.0
log_counter = 0

# Ottieni l'ID del frame
assert(model.existFrame(conf.frame_name))
frame_id = model.getFrameId(conf.frame_name)

# Precalcola la posizione iniziale dell'end-effector
pin.framesForwardKinematics(model, data, conf.qhome)
p0 = data.oMf[frame_id].translation.copy()
rpy0 = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
pitch0 = rpy0[1]

# Impostazione del logging
buffer_size = int(math.ceil(conf.exp_duration / conf.dt))
q_log = np.zeros((robot.nq, buffer_size))
p_log = np.zeros((3, buffer_size))
p_des_log = np.zeros((3, buffer_size))
pitch_log = np.zeros(buffer_size)
pitch_des_log = np.zeros(buffer_size)
time_log = np.zeros(buffer_size)

# Orientamento di pitch desiderato (in radianti)
pitch_des_final = np.radians(conf.pitch_des_deg)


# --- BLOCCO AGGIORNATO: CALCOLO DI UNA POSTURA OBIETTIVO COERENTE (versione 4D) ---
print("Calcolo di una postural target coerente tramite IK 4D...")
q_ik = conf.qhome.copy()  # Inizia dalla configurazione home

eps = 1e-4
IT_MAX = 5000
damp = 1e-5
dt_ik = 0.01 # Step size per l'IK

for i in range(IT_MAX):
    pin.forwardKinematics(model, data, q_ik)
    pin.updateFramePlacement(model, data, frame_id)
    
    # Pose attuale
    p_ik = data.oMf[frame_id].translation
    R_ik = data.oMf[frame_id].rotation
    pitch_ik = pin.rpy.matrixToRpy(R_ik)[1]
    
    # Errore 4D
    err_pos = p_ik - conf.pdes
    err_pitch = pitch_ik - pitch_des_final
    err_4d = np.hstack([err_pos, err_pitch])
    
    if np.linalg.norm(err_4d) < eps:
        print(f"IK 4D converguto con successo in {i} iterazioni.")
        break
    
    # Jacobiano 4D
    J6_ik = pin.computeFrameJacobian(model, data, q_ik, frame_id, pin.LOCAL_WORLD_ALIGNED)
    J_ik_4d = np.vstack([J6_ik[:3, :], J6_ik[4, :]])

    # Damped least-squares inverse per il task 4D
    J_inv_4d = J_ik_4d.T @ np.linalg.inv(J_ik_4d @ J_ik_4d.T + damp * np.eye(4))
    
    # Aggiorna configurazione giunti
    v_ik = -J_inv_4d @ err_4d
    q_ik = pin.integrate(model, q_ik, v_ik * dt_ik)
else: # Questo 'else' appartiene al 'for', viene eseguito se il loop finisce senza 'break'
    print("ATTENZIONE: L'IK 4D non è converguto! Il movimento potrebbe non essere corretto.")

# La configurazione trovata dall'IK diventa il nostro nuovo obiettivo posturale
q0_calibrated = q_ik.copy()
print("Nuovo postural target calibrato:", q0_calibrated)
# --- FINE BLOCCO AGGIORNATO ---


# Generatore di traiettoria per posizione e pitch
def generate_trajectory(t):
    # Posizione
    if t > conf.traj_duration:
        p_des = conf.pdes
        v_des = np.zeros(3)
        a_des = np.zeros(3)
        pitch_des = pitch_des_final
        pitch_vel_des = 0.0
        pitch_acc_des = 0.0
    else:
        p_des, v_des, a_des = np.zeros(3), np.zeros(3), np.zeros(3)
        for i in range(3):
            c_pos = coeffTraj(conf.traj_duration, p0[i], conf.pdes[i])
            p_des[i] = c_pos[0] + c_pos[1]*t + c_pos[2]*t**2 + c_pos[3]*t**3 + c_pos[4]*t**4 + c_pos[5]*t**5
            v_des[i] = c_pos[1] + 2*c_pos[2]*t + 3*c_pos[3]*t**2 + 4*c_pos[4]*t**3 + 5*c_pos[5]*t**4
            a_des[i] = 2*c_pos[2] + 6*c_pos[3]*t + 12*c_pos[4]*t**2 + 20*c_pos[5]*t**3

        # Pitch
        c_pitch = coeffTraj(conf.traj_duration, pitch0, pitch_des_final)
        pitch_des = c_pitch[0] + c_pitch[1]*t + c_pitch[2]*t**2 + c_pitch[3]*t**3 + c_pitch[4]*t**4 + c_pitch[5]*t**5
        pitch_vel_des = c_pitch[1] + 2*c_pitch[2]*t + 3*c_pitch[3]*t**2 + 4*c_pitch[4]*t**3 + 5*c_pitch[5]*t**4
        pitch_acc_des = 2*c_pitch[2] + 6*c_pitch[3]*t + 12*c_pitch[4]*t**2 + 20*c_pos[5]*t**3

    return p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des

# Loop di controllo principale
print("Avvio del controllo 4D nello spazio operativo...")
while time < conf.exp_duration:
    if log_counter >= buffer_size:
        break
    
    # Ottieni la traiettoria di riferimento
    p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des = generate_trajectory(time)
    
    # Calcola lo stato attuale
    pin.forwardKinematics(model, data, q, qd)
    pin.updateFramePlacement(model, data, frame_id)
    
    # Jacobiano 6D completo
    J6 = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    
    # Posizione e orientamento attuali
    p = data.oMf[frame_id].translation
    w_R_e = data.oMf[frame_id].rotation
    
    # Calcola il pitch attuale dagli angoli RPY
    rpy = pin.rpy.matrixToRpy(w_R_e)
    pitch = rpy[1]
    
    # Velocità dell'end-effector (twist)
    twist = J6 @ qd
    v = twist[:3]
    omega = twist[3:]
    pitch_vel = omega[1] # La velocità del pitch è la componente Y di omega

    # --- Task 4D: Posizione + Pitch ---
    # Costruzione dello Jacobiano del task 4D
    J_task = np.vstack([
        J6[0:3, :],  # Prime 3 righe per la posizione (X, Y, Z)
        J6[4, :]     # Quinta riga per il pitch (rotazione intorno a Y)
    ])

    # Errore 4D
    p_error = p_des - p
    pitch_error = pitch_des - pitch
    task_error = np.hstack([p_error, pitch_error])
    
    # Errore di velocità 4D
    v_error = v_des - v
    pitch_vel_error = pitch_vel_des - pitch_vel
    task_vel_error = np.hstack([v_error, pitch_vel_error])

    # Accelerazione desiderata 4D
    a_pos_des = a_des + conf.Kd_pos @ v_error + conf.Kp_pos @ p_error
    a_pitch_des = pitch_acc_des + conf.Kd_pitch * pitch_vel_error + conf.Kp_pitch * pitch_error
    a_task_4d = np.hstack([a_pos_des, a_pitch_des])
    
    # Calcolo della dinamica
    M = pin.crba(model, data, q)
    h = pin.nonLinearEffects(model, data, q, qd)
    Minv = np.linalg.inv(M)
    
    # Derivata dello Jacobiano
    Jdot_6d = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    Jdot_task_4d = np.vstack([Jdot_6d[0:3, :], Jdot_6d[4, :]])
    Jdotqd_task = Jdot_task_4d @ qd

    # Dinamica inversa nello spazio del task 4D
    Lambda_task = np.linalg.inv(J_task @ Minv @ J_task.T + 1e-6*np.eye(4))
    Jbar_task = Minv @ J_task.T @ Lambda_task
    
    # Proiettore nel nullo spazio del task 4D
    N_task = np.eye(robot.nv) - J_task.T @ np.linalg.pinv(J_task.T, 1e-4)
    
    # Task secondario: Posturale (usa la configurazione calibrata)
    q_postural = conf.Kp_postural * (q0_calibrated - q) - conf.Kd_postural * qd
    
    # Accelerazione desiderata combinata
    qdd_des = Jbar_task @ (a_task_4d - Jdotqd_task) + N_task @ q_postural
    
    # Coppia di comando
    tau = M @ qdd_des + h
    
    # Simulazione della dinamica in avanti (integrazione)
    qdd = Minv @ (tau - h)
    qd_next = qd + qdd * conf.dt
    q_next = pin.integrate(model, q, (qd + qd_next) * 0.5 * conf.dt)
    
    # Aggiorna stato
    q = q_next
    qd = qd_next
    
    # Log dei dati
    time_log[log_counter] = time
    q_log[:, log_counter] = q
    p_log[:, log_counter] = p
    p_des_log[:, log_counter] = p_des
    pitch_log[log_counter] = pitch
    pitch_des_log[log_counter] = pitch_des
    log_counter += 1
    
    # Aggiorna il tempo
    time += conf.dt
    
    # Visualizzazione
    ros_pub.publish(robot, q, qd, tau)
    tm.sleep(conf.dt * conf.SLOW_FACTOR)
    
    if ros_pub.isShuttingDown():
        break

# Stampa la posa finale
final_pos = data.oMf[frame_id].translation
final_orient_rpy = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
print("\nPosizione Finale dell'End Effector (m):", final_pos)
print("Orientamento Finale dell'End Effector (RPY - deg):", np.degrees(final_orient_rpy))
print("Pitch Finale (deg):", np.degrees(final_orient_rpy[1]))
print("Pitch Desiderato (deg):", np.degrees(pitch_des_final))

# Grafici dei risultati
plt.figure(figsize=(12, 10))
plt.suptitle("Performance del Controllo 4D")

# Tracking della posizione
labels_pos = ['X', 'Y', 'Z']
for i in range(3):
    plt.subplot(4, 1, i+1)
    plt.plot(time_log[:log_counter], p_log[i, :log_counter], label=f'Attuale {labels_pos[i]}')
    plt.plot(time_log[:log_counter], p_des_log[i, :log_counter], '--', label=f'Desiderato {labels_pos[i]}')
    plt.ylabel(f'Posizione {labels_pos[i]} [m]')
    plt.legend()
    plt.grid(True)

# Tracking del pitch
plt.subplot(4, 1, 4)
plt.plot(time_log[:log_counter], np.degrees(pitch_log[:log_counter]), label='Pitch Attuale')
plt.plot(time_log[:log_counter], np.degrees(pitch_des_log[:log_counter]), '--', label='Pitch Desiderato')
plt.ylabel('Pitch [deg]')
plt.xlabel('Tempo [s]')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

input("Premi Invio per terminare...")
ros_pub.deregister_node()