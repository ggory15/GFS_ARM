import numpy as np
import pinocchio as se3
import eigenpy
from scipy.spatial.transform import Rotation as R

class PiRobot:
    def __init__(self, pyRobot, dataFolder, urdfFileName, basePosition=[0,0,0], baseRPY=[0,0,0]):
        self.pyrobot = pyRobot
        self.robot = se3.RobotWrapper.BuildFromURDF(dataFolder + urdfFileName, [dataFolder])
        self.model = self.robot.model
        self.basePosition = basePosition
        self.baseRPY = baseRPY
        self.robot.q = np.zeros(self.model.nv)
        self.robot.v = np.zeros(self.model.nv)
   
    def getJointNames(self):
        try:
            return self.actuated_joint_names
        except AttributeError:
            self.actuated_joint_names = []
            for i in range(self.model.njoints-1):
                self.actuated_joint_names.append(self.model.names[i+1])
            return self.actuated_joint_names

    def getJointPositions(self):
        return dict(zip(self.getJointNames(), self.robot.q[:]))

    def updateJointStates(self):
        for name, val in self.pyrobot.getActuatedJointStates().items():         
            self.robot.q[self.model.getJointId(str(name)) - 1] = val[0]
            self.robot.v[self.model.getJointId(str(name)) - 1] = val[1]
        
    def setJointStates(self):
        self.pyrobot.setActuatedJointPositions(self.getJointPositions())
    
    def setTorqueStates(self, torque = None):
        if torque is None:
            torque = self.robot.nle(self.robot.q, self.robot.v)
        assert (len(torque) == self.robot.nv), "The number of joints is wrong."
        self.actuated_torques = dict(zip(self.getJointNames(), torque[:]))
        self.pyrobot.setActuatedJointTorques(self.actuated_torques)

    def getPlacement(self, name):
        id = self.robot.model.getFrameId(name)
        return self.robot.framePlacement(self.robot.q, id)
