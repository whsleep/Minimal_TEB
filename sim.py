import numpy as np
from irsim.env import EnvBase
from TebSolver import TebplanSolver
from irsim.lib.path_planners.a_star import AStarPlanner

from collections import namedtuple
import cv2
from sklearn.cluster import DBSCAN


obs = namedtuple('obstacle', 'center radius vertex cone_type velocity')

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):
        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal.squeeze()
        self.lidar_r = 2.0
        
        # 全局规划器
        # data = self.env.get_map()
        # self.planner = AStarPlanner(data,data.resolution)
        # self.global_path = self.planner.planning(np.array([2.0,8.0]),np.array([8.0,2.0]))
        
        # 局部求解器
        self.solver = TebplanSolver(self.robot.geometry, np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0]) )

        # 速度指令
        self.v = 0.2
        self.w = 0.2
        
    def step(self, lin_velocity=0.2, ang_velocity=0.0):
        # 环境单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()

        # 获取机器人姿态及环境信息
        robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()
        obs_list = self.scan_box(robot_state,scan_data)
        
        # 绘制障碍
        for obs in obs_list:
            print(self.robot_footprint.calculate_distance(robot_state, obs))
            self.env.draw_box(obs, refresh=True)

        # 计算临时目标点
        current_goal = self.compute_currentGoal(robot_state.squeeze())

        # # 求解局部最优轨迹
        # traj, dt_seg = self.solver.solve(robot_state.squeeze(), current_goal, pointobstacles)
        # traj_xy = traj[:, :2]         

        # # 轨迹可视化
        # traj_list = [np.array([[xy[0]], [xy[1]]]) for xy in traj_xy]
        # self.env.draw_trajectory(traj_list, 'r--', refresh=True)

        # # 计算速度指令作为下次仿真输入
        # self.compute_v_omega(traj[0,:] ,traj[1,:], dt_seg[0])


        # 是否抵达
        if self.env.robot.arrive:
            print("Goal reached")
            return True
        
        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True
        
        return False
    
    def compute_v_omega(self, p0, p1, dt):
        x0, y0, th0 = p0
        x1, y1, th1 = p1
        # 1) 线速度（带方向）
        dx = x1 - x0
        dy = y1 - y0
        v = (dx * np.cos(th0) + dy * np.sin(th0)) / dt  # 沿当前朝向的投影速度
        # 2) 角速度（带方向）
        dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))  # 最短方向 [-π, π]
        w = dth / dt
        self.v = v 
        self.w = w

    def compute_currentGoal(self, robot_state):
        # 提取位置坐标
        rx, ry = robot_state[0], robot_state[1]
        gx, gy = self.robot_goal[0], self.robot_goal[1]
        
        # 计算机器人到目标的距离
        distance = np.sqrt((gx - rx)**2 + (gy - ry)**2)
        
        # 如果目标在圆内，直接返回目标点
        if distance <= self.lidar_r:
            return self.robot_goal.squeeze()
        
        # 计算交点（临时目标点）
        t = (self.lidar_r ) / distance
        temp_x = rx + t * (gx - rx)
        temp_y = ry + t * (gy - ry)
        
        # 保持原朝向或重新计算朝向（此处保持原朝向）
        # 计算朝向全局目标的角度
        temp_theta = np.arctan2(gy - ry, gx - rx)

        
        return np.array([temp_x, temp_y, temp_theta])
    
    def scan_box(self, state, scan_data):

        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

        point_list = []
        obstacle_list = []

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < ( scan_data['range_max'] - 0.01):
                point = np.array([ [scan_range * np.cos(angle)], [scan_range * np.sin(angle)]  ])
                point_list.append(point)

        if len(point_list) < 4:
            return obstacle_list

        else:
            point_array = np.hstack(point_list).T
            labels = DBSCAN(eps=0.4, min_samples=4).fit_predict(point_array)

            for label in np.unique(labels):
                if label == -1:
                    continue
                else:
                    point_array2 = point_array[labels == label]
                    rect = cv2.minAreaRect(point_array2.astype(np.float32))
                    box = cv2.boxPoints(rect)

                    vertices = box.T

                    trans = state[0:2]
                    rot = state[2, 0]
                    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                    global_vertices = trans + R @ vertices

                    obstacle_list.append(global_vertices)

            return obstacle_list