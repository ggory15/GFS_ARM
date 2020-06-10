import numpy as np
import pybullet
import pybullet_data

from env import sim_env as pyEnv
from env import sim_robot as pyRobot
from env import pi_robot as piRobot

import pinocchio as se3

if __name__ == '__main__':
    sim_rate = 1000
    g = 9.81
    dataFolder = "/home/ggory15/catkin_ws/src/cogimon"
    urdf_filename = '/cogimon_urdf/urdf/cogimon_arm.urdf'
    base_position=[0, 0.0,1.025]
    base_RPY=[0, 0, 0]

    homing_config = [np.deg2rad(45.0), np.deg2rad(10.0), 0.0, np.deg2rad(-110.0), 0.0, np.deg2rad(-30.0), 10 ]
    pyenv = pyEnv.SimEnv(sim_rate=sim_rate)

    pyrobot = pyRobot.SimRobot(urdfFileName=dataFolder + urdf_filename,
                     basePosition=[0.0, 0, 1.5],
                     baseRPY=[0, 0, 0],
                     useFixedBase=True,
                     jointPositions=homing_config)
    pirobot = piRobot.PiRobot(pyrobot, dataFolder,  urdf_filename,  base_position, base_RPY)


    pyenv.reset()
    i = 0
    while True:
        pirobot.updateJointStates()
        if (i < 10):
            pyenv.step()
            pirobot.setJointStates()
            pyenv.debug()
        else:
            pyenv.step()
            pirobot.setTorqueStates()
            pyenv.debug()            
        
        i += 1
