import numpy as np
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase
from robot_footprint_model import CircularRobotFootprint
from obstacles import PointObstacle

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):
        # Initialize environment
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        self.robot_goal = self.env.get_robot_info(0).goal

        # get radius 
        self.robot_info = self.env.get_robot_info()
        
        self.robot_footprint = CircularRobotFootprint(self.robot_info.wheelbase)
        
    def step(self, lin_velocity=0.2, ang_velocity=0.0):
        # Step simulation
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        if self.env.display:
            self.env.render()

        self.env.get_map()
        # Get sensor data
        scan = self.env.get_lidar_scan()
        robot_state = self.env.get_robot_state()
    
        
        # Get status and calculate reward
        goal = self.env.robot.arrive
        collision = self.env.robot.collision

        if goal:
            print("Goal reached")
