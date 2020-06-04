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

    qa = np.matrix(np.zeros((7, 1)))
    qa_dot = np.matrix(np.zeros((7, 1)))

    for i in range(7):
        qa[i, 0] = robot.q[i]
        qa_dot[i, 0] = robot.v[i] 

    # Method : Joint Control A q_ddot = torque
    Kp = 400. # Convergence gain
    ID_EE = robot.model.getFrameId("LWrMot3") # Control target (End effector) ID

    # Compute/update all joints and frames
    se3.computeAllTerms(robot.model, robot.data, qa, qa_dot)

    # Get kinematics information
    oMi = robot.framePlacement(qa, ID_EE) ## EE's placement
    v_frame = robot.frameVelocity(qa, qa_dot, ID_EE) # EE's velocity
    J = robot.computeFrameJacobian(qa, ID_EE)[:3, :] ## EE's Jacobian w.r.t Local Frame

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
    p_vec = p_error.linear
    v_vec = v_error.linear

    xddotstar = (- Kp * p_vec - 2* np.sqrt(Kp) * v_vec)
    f_star = Lam*xddotstar

    # Torque from Joint error
    torque = J.transpose() * f_star + NLE    
    torque[0] = 0.0

    # Selection Matrix
    Id7 = np.identity(7)
    fault_joint_num = 0
    S = np.delete(Id7, (fault_joint_num), axis=0) #delete fault joint corresponding row
    # Selection Matrix Inverse - dynamically consistent inverse
    Minv = np.linalg.inv(M)
    ST = S.transpose()
    SbarT = np.linalg.pinv( S * Minv * ST ) * S * Minv
    # Jacobian Matrix Inverse - dynamically consistent inverse
    JbarT = Lam* J * np.linalg.inv(M)
    #Weighting matrix
    W = S * Minv * ST
    Winv = np.linalg.inv(W)
    #Weighted pseudo inverse of JbarT*ST
    JtildeT = Winv * (JbarT*ST).transpose() * np.linalg.pinv(JbarT*ST * Winv * (JbarT*ST).transpose())
    #Null-space projection matrix
    Id6 = np.identity(6)
    NtildeT = Id6 - JtildeT*JbarT*ST
    null_torque = 0.0 * qa_dot  # joint damping

    #Torque
    torque = ST* (JtildeT*f_star + NtildeT*SbarT*null_torque + SbarT*NLE)
    
    return torque


