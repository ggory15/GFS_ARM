import numpy as np
# Pinocchio modules
import pinocchio as se3  # Pinocchio library
import eigenpy
eigenpy.switchToNumpyMatrix()

import copy

################
#   PlANNERL  ##
################

def c_planner(dt, traj, oMi, oMi_init, t_simu):
    eigenpy.switchToNumpyMatrix()
    EPS = 1e-6
  
    # Update Target oMi position
    if np.linalg.norm(t_simu - 0.0) < EPS:
        oMi_ref = copy.deepcopy(oMi_init)
        oMi_ref.translation = oMi_ref.translation + np.matrix([0.0, 0.05, 0.05]).transpose()

        traj.setInitialPosture(oMi_init)
        traj.setFinalPosture(oMi_ref)
        traj.setDuration(1.0)
        traj.setStartTime(0.0)
    elif np.linalg.norm(t_simu - 1.0) < EPS:
        oMi_ref = copy.deepcopy(oMi_init)
        oMi_ref.translation = oMi_ref.translation + np.matrix([0.0, 0.0, 0.1]).transpose()
        traj.setInitialPosture(oMi)
        traj.setFinalPosture(oMi_ref)
        traj.setDuration(1.0)
        traj.setStartTime(1.0)
    elif np.linalg.norm(t_simu - 2.0) < EPS:
        oMi_ref = copy.deepcopy(oMi_init)
        traj.setInitialPosture(oMi)
        traj.setFinalPosture(oMi_ref)
        traj.setDuration(1.0)
        traj.setStartTime(2.0)

    sample = traj.computeNext(t_simu)
    return sample
