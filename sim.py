import numpy as np
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase
from TebSolver import TrajOpt
# from robot_footprint_model import CircularRobotFootprint
# from obstacles import PointObstacle

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):
        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        self.robot_goal = self.env.get_robot_info(0).goal

        # 机器人半径
        self.robot_radius = 0.34

        # 求解器
        self.solver = TrajOpt()

        self.v = 0.0
        self.w = 0.0
        
    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        # 单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        if self.env.display:
            self.env.render()

        # self.env.get_map()
        # 获取雷达数据
        # scan = self.env.get_lidar_scan()
        # 机器人姿态
        robot_state = self.env.get_robot_state()
        # 
        self.pointobstacles = self.env.get_obstacle_info_list()
        self.pointobstacles = [obs.center[:2].T for obs in self.pointobstacles]
        # 构建点障碍
        # self.pointobstacles = self.generate_obstacles(scan, robot_state)

        traj, dt_seg = self.solver.solve(robot_state, self.robot_goal, self.pointobstacles)
        traj_xy = traj[:, :2]          # 只取前两列 (x, y)
        traj_list = [np.array([[xy[0]], [xy[1]]]) for xy in traj_xy]

        self.env.draw_trajectory(traj_list, 'r', refresh=True)

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

    def generate_obstacles(self, scan, robot_state):
        # 1. 生成与 ranges 等长的方位角数组
        angles = np.arange(scan['angle_min'],
                        scan['angle_max'] + scan['angle_increment'] * 0.5,
                        scan['angle_increment'])
        angles = angles[:len(scan['ranges'])]

        # 2. 过滤无效/超量程测距
        ranges = np.asarray(scan['ranges'])
        valid = (ranges >= scan['range_min']) & \
                (ranges <= scan['range_max']-0.1) & \
                np.isfinite(ranges)
        angles = angles[valid]
        ranges = ranges[valid]

        # 3. 极坐标 → 车体坐标系
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # 4. 车体坐标系 → 全局坐标系
        x, y, theta = robot_state
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_world = x + cos_t * x_local - sin_t * y_local
        y_world = y + sin_t * x_local + cos_t * y_local

        # 5. 组装 PointObstacle 列表
        obstacles = [np.array([xi, yi]) for xi, yi in zip(x_world, y_world)]

        return obstacles
    
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
