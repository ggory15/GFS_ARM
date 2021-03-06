import numpy as np
import pybullet
import pybullet_data

from env import sim_env as pyEnv
from env import sim_robot as pyRobot
from env import pi_robot as piRobot

import pinocchio as se3
from OSF_w import controller, planner
import Trajectories as traj

if __name__ == '__main__':
    sim_rate = 1000
    dt = 1.0 / sim_rate
    g = 9.81
    dataFolder = "/home/user/catkin_ws/src/iit-cogimon-ros-pkg"
    urdf_filename = '/cogimon_urdf/urdf/cogimon_arm.urdf'
    base_position=[0, 0.0,1.025]
    base_RPY=[0, 0, 0]

    homing_config = [np.deg2rad(45.0), np.deg2rad(10.0), 0.0, np.deg2rad(-110.0), 0.0, np.deg2rad(-30.0), 0 ]
    pyenv = pyEnv.SimEnv(sim_rate=sim_rate)

    pyrobot = pyRobot.SimRobot(urdfFileName=dataFolder + urdf_filename,
                     basePosition=[0.0, 0, 1.5],
                     baseRPY=[0, 0, 0],
                     useFixedBase=True,
                     jointPositions=homing_config)
    pirobot = piRobot.PiRobot(pyrobot, dataFolder,  urdf_filename,  base_position, base_RPY)


    pyenv.reset()
    i = 0
    
    trajSE3 = traj.TrajectorySE3Cubic() #Trajectory for drawing triangle
    oMi_init = [] # global variable for Posture
    
    while True:
        pirobot.updateJointStates()
        
        if i == 0:
            oMi_init = pirobot.getPlacement("LWrMot3") ## EE's placement
        
        sample = planner.c_planner( dt, trajSE3, pirobot.getPlacement("LWrMot3"), oMi_init, i * dt)
        torque = controller.c_control(dt, pirobot.robot, sample, i * dt)

        pyenv.step()
        pirobot.setTorqueStates(torque)
        pyenv.debug()            
        
        i += 1