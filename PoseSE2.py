import numpy as np

class PoseSE2:
    def __init__(self, position=np.array([0.0, 0.0]), orientation=0.0):
        self.position = position
        self.orientation = orientation

    def to_pose_msg(self, pose_msg):
        pose_msg.position.x = self.position[0]
        pose_msg.position.y = self.position[1]
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = np.sin(self.orientation / 2)
        pose_msg.orientation.w = np.cos(self.orientation / 2)