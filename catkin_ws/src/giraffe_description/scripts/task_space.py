#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function

import pinocchio as pin
import numpy as np
import math
import time as tm
import matplotlib.pyplot as plt
import os

from utils.common_functions import getRobotModel
from utils.kin_dyn_utils import fifthOrderPolynomialTrajectory as coeffTraj
import conf as conf

import random

time = 0.0

def FK(model, data):
    pin.framesForwardKinematics(model, data, conf.qhome)
    p0 = data.oMf[model.getFrameId(conf.frame_name)].translation.copy()
    rpy0 = pin.rpy.matrixToRpy(data.oMf[model.getFrameId(conf.frame_name)].rotation)
    pitch0 = rpy0[1]

    return p0, rpy0, pitch0

def log_initialize(robot):
    buffer_size = int(math.ceil(conf.exp_duration / conf.dt))
    q_log = np.zeros((robot.nq, buffer_size))
    p_log = np.zeros((3, buffer_size))
    p_des_log = np.zeros((3, buffer_size))
    pitch_log = np.zeros(buffer_size)
    pitch_des_log = np.zeros(buffer_size)
    time_log = np.zeros(buffer_size)

    return buffer_size, q_log, p_log, p_des_log, pitch_log, pitch_des_log, time_log

def postural_target_computation(model, data, pitch_des_final):
    eps = 1e-4
    IT_MAX = 5000
    damp = 1e-5
    dt_ik = 0.01 # Step size per l'IK
    q_ik = conf.qhome.copy()

    for i in range(IT_MAX):
        pin.forwardKinematics(model, data, q_ik)
        pin.updateFramePlacement(model, data, model.getFrameId(conf.frame_name))

        # Pose attuale
        p_ik = data.oMf[model.getFrameId(conf.frame_name)].translation
        R_ik = data.oMf[model.getFrameId(conf.frame_name)].rotation
        pitch_ik = pin.rpy.matrixToRpy(R_ik)[1]

        # Errore 4D
        err_pos = p_ik - conf.pdes
        err_pitch = pitch_ik - pitch_des_final
        err_4d = np.hstack([err_pos, err_pitch])

        if np.linalg.norm(err_4d) < eps:
            print(f"IK 4D converged in {i} iterations.")
            break

        # Jacobiano 4D
        J6_ik = pin.computeFrameJacobian(model, data, q_ik, model.getFrameId(conf.frame_name), pin.LOCAL_WORLD_ALIGNED)
        J_ik_4d = np.vstack([J6_ik[:3, :], J6_ik[4, :]])

        # Damped least-squares inverse per il task 4D
        J_inv_4d = J_ik_4d.T @ np.linalg.inv(J_ik_4d @ J_ik_4d.T + damp * np.eye(4))

        # Aggiorna configurazione giunti
        v_ik = -J_inv_4d @ err_4d
        q_ik = pin.integrate(model, q_ik, v_ik * dt_ik)
    else:
        print("IK doesn't converges.")
    q0_calibrated = q_ik.copy()
    print("New postural task calibrated: ", q0_calibrated)

    return q0_calibrated

def generate_trajectory(t, pitch_des_final, model, data):
    p0, _, pitch0 = FK(model, data)
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

        c_pitch = coeffTraj(conf.traj_duration, pitch0, pitch_des_final)
        pitch_des = c_pitch[0] + c_pitch[1]*t + c_pitch[2]*t**2 + c_pitch[3]*t**3 + c_pitch[4]*t**4 + c_pitch[5]*t**5
        pitch_vel_des = c_pitch[1] + 2*c_pitch[2]*t + 3*c_pitch[3]*t**2 + 4*c_pitch[4]*t**3 + 5*c_pitch[5]*t**4
        pitch_acc_des = 2*c_pitch[2] + 6*c_pitch[3]*t + 12*c_pitch[4]*t**2 + 20*c_pitch[5]*t**3

    return p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des

def simulation(ros_pub, robot, model, data, pitch_des_final, q, qd):
    time = 0.0
    log_counter = 0
    buffer_size, q_log, p_log, p_des_log, pitch_log, pitch_des_log, time_log = log_initialize(robot)
    q0_calibrated = postural_target_computation(model, data, pitch_des_final)
    frame_id = model.getFrameId(conf.frame_name)

    while time < conf.exp_duration:
        if log_counter >= buffer_size:
            break

        p_des, v_des, a_des, pitch_des, pitch_vel_des, pitch_acc_des = generate_trajectory(time, pitch_des_final, model, data)

        pin.forwardKinematics(model, data, q, qd)
        pin.updateFramePlacement(model, data, frame_id)

        J6 = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)

        p = data.oMf[frame_id].translation
        w_R_e = data.oMf[frame_id].rotation

        rpy = pin.rpy.matrixToRpy(w_R_e)
        pitch = rpy[1]

        twist = J6 @ qd
        v = twist[:3]
        omega = twist[3:]
        pitch_vel = omega[1]

        J_task = np.vstack([
            J6[0:3, :],
            J6[4, :]
        ])

        p_error = p_des - p
        pitch_error = pitch_des - pitch
        task_error = np.hstack([p_error, pitch_error])

        v_error = v_des - v
        pitch_vel_error = pitch_vel_des - pitch_vel
        task_vel_error = np.hstack([v_error, pitch_vel_error])

        a_pos_des = a_des + conf.Kd_pos @ v_error + conf.Kp_pos @ p_error
        a_pitch_des = pitch_acc_des + conf.Kd_pitch * pitch_vel_error + conf.Kp_pitch * pitch_error
        a_task_4d = np.hstack([a_pos_des, a_pitch_des])

        M = pin.crba(model, data, q)
        h = pin.nonLinearEffects(model, data, q, qd)
        Minv = np.linalg.inv(M)

        Jdot_6d = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        Jdot_task_4d = np.vstack([Jdot_6d[0:3, :], Jdot_6d[4, :]])
        Jdotqd_task = Jdot_task_4d @ qd

        Lambda_task = np.linalg.inv(J_task @ Minv @ J_task.T + 1e-6*np.eye(4))
        Jbar_task = Minv @ J_task.T @ Lambda_task

        N_task = np.eye(robot.nv) - J_task.T @ np.linalg.pinv(J_task.T, 1e-4)

        q_postural = conf.Kp_postural * (q0_calibrated - q) - conf.Kd_postural * qd

        qdd_des = Jbar_task @ (a_task_4d - Jdotqd_task) + N_task @ q_postural

        tau = M @ qdd_des + h

        qdd = Minv @ (tau - h)
        qd_next = qd + qdd * conf.dt
        q_next = pin.integrate(model, q, (qd + qd_next) * 0.5 * conf.dt)

        q = q_next
        qd = qd_next

        time_log[log_counter] = time
        q_log[:, log_counter] = q
        p_log[:, log_counter] = p
        p_des_log[:, log_counter] = p_des
        pitch_log[log_counter] = pitch
        pitch_des_log[log_counter] = pitch_des
        log_counter += 1

        time += conf.dt

        ros_pub.publish(robot, q, qd, tau)
        tm.sleep(conf.dt * conf.SLOW_FACTOR)

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Performance del Controllo 4D")

    labels_pos = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(4, 1, i+1)
        plt.plot(time_log[:log_counter], p_log[i, :log_counter], label=f'Attuale {labels_pos[i]}')
        plt.plot(time_log[:log_counter], p_des_log[i, :log_counter], '--', label=f'Desiderato {labels_pos[i]}')
        plt.ylabel(f'Posizione {labels_pos[i]} [m]')
        plt.legend()
        plt.grid(True)

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
    # Restituisce le configurazioni finali dei giunti
    return q, qd