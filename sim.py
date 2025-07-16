import numpy as np
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase
from TebSolver import TebplanSolver

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):
        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        self.robot_goal = self.env.get_robot_info(0).goal.squeeze()

        # 机器人半径
        self.robot_radius = 0.34

        # 雷达半径
        self.lidar_r = 4.0
        
        # 求解器
        self.solver = TebplanSolver(np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0]) )

        self.v = 0.0
        self.w = 0.0
        
    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        # 单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        if self.env.display:
            self.env.render()

        # 机器人姿态
        robot_state = self.env.get_robot_state()
        # 获取障碍
        pointobstacles = self.env.get_obstacle_info_list()
        pointobstacles = [obs.center[:2].T for obs in pointobstacles]
        pointobstacles = self.filter_obstacles_by_distance(robot_state.squeeze(), pointobstacles)
        
        # 计算临时目标点
        current_goal = self.compute_currentGoal(robot_state.squeeze())

        traj, dt_seg = self.solver.solve(pointobstacles, robot_state.squeeze(), current_goal)
        traj_xy = traj[:, :2]          # 只取前两列 (x, y)
        traj_list = [np.array([[xy[0]], [xy[1]]]) for xy in traj_xy]

        self.env.draw_trajectory(traj_list, 'r--', refresh=True)

        self.compute_v_omega(traj[0,:] ,traj[1,:], dt_seg[0])


        # 获取是否抵达目标的状态位
        goal = self.env.robot.arrive
        # 获取是否碰撞标志位
        collision = self.env.robot.collision


        if goal:
            print("Goal reached")
            return True
        if collision:
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
        if distance <= self.lidar_r-2.0:
            return self.robot_goal.squeeze()
        
        # 计算交点（临时目标点）
        t = (self.lidar_r - 3.0) / distance
        temp_x = rx + t * (gx - rx)
        temp_y = ry + t * (gy - ry)
        
        # 保持原朝向或重新计算朝向（此处保持原朝向）
        # 计算朝向全局目标的角度
        temp_theta = np.arctan2(gy - ry, gx - rx)

        
        return np.array([temp_x, temp_y, temp_theta])
    
    # def obstacle_filter(self, obstacles):
    def filter_obstacles_by_distance(self, robot_state, obstacles):
        robot_pos = robot_state[:2]  # 提取机器人位置 [x, y]
        
        # 计算每个障碍物到机器人的距离，并过滤
        filtered = []
        for obs in obstacles:
            obs_pos = obs[0]  # 提取障碍物位置 (形状为 (2,) 的数组)
            distance = np.linalg.norm(obs_pos - robot_pos)
            if distance <= self.lidar_r:
                filtered.append(obs)
        
        return filtered