#!/usr/bin/env python
from __future__ import print_function

import numpy as np # Assicurati che numpy sia importato
import matplotlib.pyplot as plt # Assicurati che matplotlib.pyplot sia importato come plt
import pinocchio as pin # Assicurati che pinocchio sia importato

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.in_kin_pinocchio import robotKinematics
from utils.math_tools import Math

from kinematics import * # Contiene dk_test e jacobian_test
from task_space import simulation # Contiene simulation (decommentata come richiesto)
from dynamics import dyn_simulation # Contiene dyn_simulation

import conf as conf

# Inizializza RosPub una sola volta all'inizio
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

# --- INIZIO DELLA LOGICA DI SCELTA DA TERMINALE ---

print("\n--- Scegli il tipo di test/simulazione da eseguire ---")
print("1: Test Cinematica Diretta e Jacobiano (kinematics)")
print("2: Simulazione Dinamica (dynamics)")
print("3: Simulazione Task Space (simulation)")
print("4: Visualizzazione Robot (RViz Only)")
choice = input("Inserisci il numero corrispondente al test/simulazione desiderato: ")

# Variabili per i log dei plot (inizializzate a None, verranno riempite solo se scelta l'opzione 'dynamics')
time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log = \
    None, None, None, None, None, None, None, None

if choice == '1':
    print("\nEsecuzione: Test Cinematica Diretta e Jacobiano")
    #####################################################################
    # Test Direct Kinematics
    #####################################################################
    dk_test(q, qd, robot, frame_id)
    jacobian_test(q, frame_id, robot)
    # Questo test non produce i log di tempo/posizione ecc. per i plot finali.

elif choice == '2':
    print("\nEsecuzione: Simulazione Dinamica")
    #####################################################################
    # Test Dynamics
    #####################################################################
    ros_pub = RosPub("giraffe_robot")
    time_log, q_log, qd_log, qdd_log, q_des_log, qd_des_log, qdd_des_log, tau_log = \
        dyn_simulation(robot, time, ros_pub, q, qd, qdd, q_des, qd_des, qdd_des)

elif choice == '3':
    print("\nEsecuzione: Simulazione Task Space")
    #####################################################################
    # Test Simulation (Task Space)
    #####################################################################
    ros_pub = RosPub("giraffe_robot")
    
    q_sim = conf.qhome.copy()
    qd_sim = np.zeros(robot.nv)
    pitch_des_final = np.radians(conf.pitch_des_deg)

    # La funzione simulation in task_space.py gestisce i propri plot internamente.
    # Se volessi spostare anche quei plot qui, dovresti modificare task_space.py
    # per fargli restituire i log, in modo simile a dyn_simulation.
    q_final, qd_final = simulation(ros_pub, robot, model, data, pitch_des_final, q_sim, qd_sim)

    # Aggiorna esplicitamente la cinematica diretta del robot con i valori finali
    pin.forwardKinematics(model, data, q_final, qd_final)
    pin.updateFramePlacement(model, data, frame_id)

    # Stampa la posa finale usando i dati aggiornati
    final_pos = data.oMf[frame_id].translation
    final_orient_rpy = pin.rpy.matrixToRpy(data.oMf[frame_id].rotation)
    print("\nPosizione Finale dell'End Effector (m):", final_pos)
    print("Orientamento Finale dell'End Effector (RPY - deg):", np.degrees(final_orient_rpy))
    print("Pitch Finale (deg):", np.degrees(final_orient_rpy[1]))
    print("Pitch Desiderato (deg):", np.degrees(pitch_des_final))

elif choice == '4': # Nuova opzione per la sola visualizzazione
    print("\nEsecuzione: Visualizzazione Robot in RViz")
    #####################################################################
    # Visualizzazione Robot (RViz Only)
    #####################################################################
    # Inizializza RosPub per avviare RViz
    ros_pub = RosPub("giraffe_robot")

    # Pubblica la configurazione iniziale del robot per mostrarlo in RViz
    # Puoi usare conf.q0 o una configurazione specifica per la visualizzazione
    ros_pub.publish(robot, conf.q0, np.zeros(robot.nv)) # Pubblica q e qd=0

    print("RViz avviato. Premi Ctrl+C nel terminale per terminare la visualizzazione.")
    # Mantiene il nodo ROS attivo per mantenere la finestra di RViz aperta
    while ros_pub is not None and not ros_pub.isShuttingDown():
        try:
            ros.sleep(1.0) # Dorme per 1 secondo per non sprecare CPU
        except ros.ROSInterruptException:
            # Cattura l'eccezione se Ctrl+C viene premuto durante ros.sleep()
            break
    print("Visualizzazione RViz terminata.")

else:
    print("Scelta non valida. Nessun test/simulazione verrà eseguito.")

# --- FINE DELLA LOGICA DI SCELTA DA TERMINALE ---

# Deregistra il nodo ROS (deve essere chiamato una sola volta alla fine del programma)
# Ho rimosso la deregistrazione che avevi dopo la chiamata a dyn_simulation.
if ros_pub is not None:
    ros_pub.deregister_node()

#####################################################################
# Genera e Mostra i Plot Finali (solo se i log sono disponibili)
#####################################################################
# Plot only for dynamics simulation
if time_log is not None and len(time_log) > 0:
    print("\nGenerazione dei plot...")
    plt.figure(1)
    plotJoint('position', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(2)
    plotJoint('velocity', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(3)
    plotJoint('acceleration', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    plt.figure(4)
    plotJoint('torque', time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log, tau_log)
    
    # Mostra tutti i plot e aspetta che l'utente li chiuda manualmente
    plt.show()

    input("Premi Invio per terminare i plot...")

input("Programma terminato. Premi Invio per uscire.")