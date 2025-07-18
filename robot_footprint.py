from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform


class BaseRobotFootprintModel(ABC):
    """抽象接口：所有 footprint 模型必须实现这两个方法。"""
    @abstractmethod
    def calculate_distance(self, current_pose, obstacle) -> float:
        """计算机器人在 current_pose 到 obstacle 的最短距离（>0 表示无碰撞）。"""
        pass

    @abstractmethod
    def check_collision(self, current_pose, obstacle ,min_dist: float) -> bool:
        """如果机器人轮廓与障碍物距离小于 min_dist, 则认为发生碰撞。"""
        pass


class RobotPolygonFootprint(BaseRobotFootprintModel):
    """
    多边形机器人 footprint。

    参数
    ----
    local_vertices : Sequence[Tuple[float, float]]
        以机器人中心为原点的本地坐标系下的多边形顶点，顺序需满足右手规则（逆时针）。
    """

    def __init__(self, local_vertices: Sequence[Tuple[float, float]]):
        super().__init__()
        # 构造局部多边形
        self.local_polygon: Polygon = Polygon(local_vertices)
        if not self.local_polygon.is_valid:
            raise ValueError("提供的顶点无法构成合法多边形")

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #
    def calculate_distance(self, current_pose, obstacle) -> float:
        """
        计算机器人 footprint 到障碍物的最短距离。

        参数
        ----
        current_pose : tuple or array_like
            [x, y, yaw] 机器人在世界坐标系下的位姿。
        obstacle : shapely.geometry object or (N,2) array
            障碍物，可以是 shapely 对象，也可以是顶点数组，内部会转成 Polygon。

        返回
        ----
        float
            最短距离。若轮廓与障碍物相交，则返回负值（距离绝对值 = 最小穿透深度）。
        """
        world_polygon = self._transform_to_world(current_pose)
        obs_polygon   = self._ensure_polygon(obstacle.T)

        # 最短距离：两个几何体之间的最小距离
        dist = world_polygon.distance(obs_polygon)

        # 若相交，distance 返回 0，此时用 intersection 计算穿透深度
        if world_polygon.intersects(obs_polygon):
            penetration = world_polygon.intersection(obs_polygon).area ** 0.5  # 近似
            return -penetration
        return dist

    def check_collision(self, current_pose, obstacle, min_dist: float) -> bool:
        """
        判断机器人是否在给定安全距离内与障碍物碰撞。
        """
        return self.calculate_distance(current_pose, obstacle) < min_dist

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_polygon(obstacle):
        """把障碍物统一转成 Polygon。"""
        if isinstance(obstacle, Polygon):
            return obstacle
        # 支持顶点数组
        return Polygon(np.asarray(obstacle))

    def _transform_to_world(self, pose) -> Polygon:
        """
        将局部 footprint 变换到世界坐标系。

        pose : array_like
            [x, y, yaw]
        """
        x, y, yaw = pose
        cos, sin = np.cos(yaw), np.sin(yaw)
        # 仿射变换矩阵 [a, b, d, e, xoff, yoff]
        mat = [cos, -sin, sin, cos, x, y]
        return affine_transform(self.local_polygon, mat)
    

if __name__ == "__main__":
    # 构造一个 1m × 0.8m 的矩形机器人
    robot = RobotPolygonFootprint([
        (-0.5, -0.4),
        ( 0.5, -0.4),
        ( 0.5,  0.4),
        (-0.5,  0.4)
    ])

    # 障碍物（同样用多边形表示）
    obstacle = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

    pose = (0, 0, 0)           # 机器人在原点
    print("最短距离:", robot.calculate_distance(pose, obstacle))
    print("是否碰撞 (阈值 0.5):", robot.check_collision(pose, obstacle, 0.5))