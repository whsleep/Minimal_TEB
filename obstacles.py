from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import distance

class Obstacle(ABC):
    def __init__(self, dynamic=False, centroid_velocity=np.array([0.0, 0.0])):
        super().__init__() 
        self.dynamic = dynamic
        self.centroid_velocity = centroid_velocity

    @abstractmethod
    def get_centroid(self):
        pass

    @abstractmethod
    def check_collision(self, position, min_dist):
        pass

    @abstractmethod
    def get_minimum_distance(self, position):
        pass

    def is_dynamic(self):
        return self.dynamic

    def set_centroid_velocity(self, vel):
        self.centroid_velocity = vel
        self.dynamic = True

    def get_centroid_velocity(self):
        return self.centroid_velocity

        
class PointObstacle(Obstacle):
    def __init__(self, position=np.array([0.0, 0.0])):
        super().__init__()
        self.position = position

    def get_centroid(self):
        return self.position

    def check_collision(self, position, min_dist):
        return self.get_minimum_distance(position) < min_dist

    def get_minimum_distance(self, position):
        return distance.euclidean(self.position, position)
