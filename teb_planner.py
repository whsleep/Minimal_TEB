
import numpy as np
import PoseSE2
import g2o


class teb_planner():
    def __init__(self):
        self.initialized_ = False
        self.optimized_ = False

        self.optimizer = self.initOptimizer()

    def initOptimizer(self):
        self.initialized_ = True
        return True

    def initTrajectoryToGoal(self ):
        return True
    