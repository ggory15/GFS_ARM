import numpy as np
import pinocchio as se3
from .trajectory_base import TrajectoryBase, TrajectorySample
import copy

class TrajectorySE3Cubic(TrajectoryBase):
    def __init__(self, name=None, M_init=None, M_goal=None, duration=None, stime=None):
        self.name = name
        self.M_init = M_init
        self.M_goal = M_goal
        self.duration = duration
        self.stime = stime
        
        self.m_sample = TrajectorySample(12, 6)
        self.m_sample.setPos(self.M_init) 
        
    def size(self):
        return 6

    def computeNext(self, time):
        if time < self.stime:
            self.m_sample.setPos(self.M_init) 
        elif time >=self.stime and time < self.stime + self.duration:
            M = se3.SE3()
            M = copy.deepcopy(self.M_init)

            dt = time - self.stime
            dx = self.M_goal.translation- self.M_init.translation     
            pos = np.matrix(np.zeros((3,1)))
            for i in range(0, 3):
                pos[i] =  - 2.0 * dx[i] / pow(self.duration, 3) * pow(dt, 3) + 3.0 * dx[i] / pow(self.duration, 2) * pow(dt, 2) + self.M_init.translation[i]

            M.translation = pos
            self.m_sample.setPos(M) 
        else:
            self.m_sample.setPos(self.M_goal) 

        return self.m_sample

    def getLastSample(self):
        self.m_sample.setPos(self.M_goal)
        return self.m_sample

    def has_trajectory_ended(self):
        return True

    def setInitialPosture(self, M_init):
        self.M_init = M_init
    
    def setDuration(self, duration):
        self.duration = duration

    def setFinalPosture(self, M_goal):
        self.M_goal = M_goal

    def setStartTime(self, stime):
        self.stime = stime