import numpy as np
# Pinocchio modules
import pinocchio as se3  # Pinocchio library
import eigenpy
eigenpy.switchToNumpyMatrix()

import copy


################
#  CONTROLLER ##
################

def c_control(dt, robot, sample, t_simu):
    eigenpy.switchToNumpyMatrix()

    qa = robot.q # actuation joint position
    qa_dot = robot.v # actuation joint velocity

    # Method : Joint Control A q_ddot = torque
    Kp = 400. # Convergence gain
    ID_EE = robot.model.getFrameId("LWrMot3") # Control target (End effector) ID

    # Compute/update all joints and frames
    se3.computeAllTerms(robot.model, robot.data, qa, qa_dot)

    # Get kinematics information
    oMi = robot.framePlacement(qa, ID_EE) ## EE's placement
    v_frame = robot.frameVelocity(qa, qa_dot, ID_EE) # EE's velocity
    J = robot.computeFrameJacobian(qa, ID_EE) ## EE's Jacobian w.r.t Local Frame

    # Get dynamics information
    M = robot.mass(qa) # Mass Matrix
    NLE = robot.nle(qa, qa_dot) #gravity and Coriolis
    Lam = np.linalg.inv( J * np.linalg.inv(M) * J.transpose() ) # Lambda Matrix

    # Update Target oMi position
    wMl = copy.deepcopy(sample.pos)
    wMl.rotation = copy.deepcopy(oMi.rotation)
    v_frame_ref = se3.Motion() # target velocity (v, w)
    v_frame_ref.setZero()

    # Get placement Error
    p_error = se3.log(sample.pos.inverse() * oMi )
    v_error = v_frame - wMl.actInv(v_frame_ref)

    # Task Force
    p_vec = p_error.vector
    v_vec = v_error.vector

    for i in range(0, 3): # for position only control
        p_vec[i+3] = 0.0
        v_vec[i+3] = 0.0

    f_star = Lam * (- Kp * p_vec - 2.0* np.sqrt(Kp) * v_vec)
    

    # Torque from Joint error
    torque = J.transpose() * f_star + NLE    
    torque[0] = 0.0
    
    return torque
