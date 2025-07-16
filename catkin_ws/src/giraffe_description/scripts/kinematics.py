#!/usr/bin/env python
# coding=utf-8 
from __future__ import print_function

import pinocchio as pin
from pinocchio.utils import *
import numpy as np
from numpy import nan
import math
import time as tm

from utils.common_functions import *
from utils.ros_publish import RosPub
from utils.kin_dyn_utils import directKinematics
from utils.kin_dyn_utils import computeEndEffectorJacobian
from utils.kin_dyn_utils import geometric2analyticJacobian
from utils.math_tools import Math
import matplotlib.pyplot as plt

import conf as conf

# Direct kinematics
def dk_test(q, qd, robot, frame_id):
    T_wb, T_w1, T_w2, T_w3, T_w4, T_we, T_wt = directKinematics(q)
    # compare with Pinocchio built-in functions 
    robot.computeAllTerms(q, qd)
    x = robot.framePlacement(q, frame_id).translation
    o = robot.framePlacement(q, frame_id).rotation
    position_diff = x - T_wt[:3,3]
    rotation_diff = o - T_wt[:3,:3]
    print("Direct Kinematics - ee position, differece with Pinocchio library:", position_diff)
    print("Direct Kinematics - ee orientation, differece with Pinocchio library:\n", rotation_diff)

def jacobian_test(q, frame_id, robot):
    _, _, _, _, _, _, T_wt = directKinematics(q)
    # Geometric Jacobian
    J, z1, z2, z3, z4, z5 = computeEndEffectorJacobian(q)
    Jee = robot.frameJacobian(q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Computed with Pinocchio
    jacobian_diff = J - Jee
    print("\n------------------------------------------")
    np.set_printoptions(suppress=True, precision=2)
    print("Direct Kinematics - ee Gometric Jacobian (6X5 matrix), differece with Pinocchio library:\n", jacobian_diff)
    print("Geometric Jacobian:\n", J)


    # Analytic Jacobian
    J_a = geometric2analyticJacobian(J, T_wt)
    print("Analytic Jacobian:\n", J_a)











