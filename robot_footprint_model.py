from abc import ABC, abstractmethod
import numpy as np
import PoseSE2


class BaseRobotFootprintModel(ABC):
    def __init__(self):
        super().__init__() 

    @abstractmethod
    def calculate_distance(self, current_pose, obstacle):
        pass

    @abstractmethod
    def get_inscribed_radius(self):
        pass

class CircularRobotFootprint(BaseRobotFootprintModel):
    def __init__(self, radius=1.0):
        super().__init__() 
        self.radius = radius

    def set_radius(self, radius):
        self.radius = radius

    def calculate_distance(self, current_pose, obstacle):
        return obstacle.get_minimum_distance(current_pose.position) - self.radius

    def get_inscribed_radius(self):
        return self.radius