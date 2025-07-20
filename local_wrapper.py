import numpy as np

class LocalGoalWrapper:
    def __init__(self, alpha=1.0, beta=0.7):
        """
        初始化局部目标包裹器
        
        参数:
            alpha: 评价函数中角度区间长度的权重
            beta: 评价函数中角度接近目标方向的权重
        """
        self.alpha = alpha
        self.beta = beta
        
    def compute_free_sectors(self, scan_data, obstacle_threshold_ratio=0.4, min_consecutive=6):
        ranges = np.array(scan_data['ranges'])
        angle_min = float(scan_data['angle_min'])
        angle_max = float(scan_data['angle_max'])
        range_max = float(scan_data['range_max'])

        angles = np.linspace(angle_min, angle_max, len(ranges))

        # True 表示“可通行”
        mask = ranges > range_max * obstacle_threshold_ratio

        # 找到所有连续 True 的 (start, stop) 区间
        d = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]

        # 只保留长度 ≥ min_consecutive 的区间
        ok = (ends - starts) >= min_consecutive
        starts, ends = starts[ok], ends[ok]

        # 展开区间，提取所有对应角度
        idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        return angles[idx]
        
       
    
    def evaluate_free_sectors(self, free_sectors, target_angle, current_angle):
        """
        评价可行角度区间，选择最优区间索引
        步骤：
            1) 计算原始差值
            2) 按区间内的最大差值统一归一化（0~1）
            3) 加权求分，取最小分对应的索引
        返回：
            best_sector_idx : 最优区间的索引，若 free_sectors 为空则返回 -1
        """

        # 1) 先把列表转成 numpy 数组，后续向量化方便
        angles = np.asarray(free_sectors, dtype=float)

        # 2) 计算原始差值向量
        d_target = np.abs(angles - target_angle)
        d_current = np.abs(angles - current_angle)

        # 3) 计算归一化分母：用区间内的最大差值，避免除 0
        max_diff = max(np.max(d_target), np.max(d_current)) + 1e-9

        # 4) 归一化
        norm_target = d_target / max_diff
        norm_current = d_current / max_diff

        # 5) 加权得分
        scores = self.alpha * norm_target + self.beta * norm_current

        # 6) 返回最小分对应的索引
        best_sector_idx = int(np.argmin(scores))
        return best_sector_idx
    
    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def generate_local_goal(self, robot_state, temp_goal, scan_data):
        """生成局部目标点
        
        参数:
            robot_state: 机器人状态 [x, y, theta]
            temp_goal: 临时目标点 [x_i, y_i]
            scan_data: 激光雷达数据
        """
        robot_state = robot_state.T
        robot_state = robot_state[0, :3].squeeze()
        temp_goal = temp_goal.T
        temp_goal = temp_goal[0, :2].squeeze()
        # 提取机器人位置
        rx, ry = robot_state[0], robot_state[1]
        
        # 计算目标距离和方向
        l_goal = np.linalg.norm(temp_goal - np.array([rx, ry]))
        target_angle = np.arctan2(temp_goal[1] - ry, temp_goal[0] - rx)
        
        # 计算可行角度
        free_sectors = self.compute_free_sectors(scan_data)
        
        # 评价可行角度
        best_sector = self.evaluate_free_sectors(free_sectors, target_angle, robot_state[-1])
        
        goal_angle = free_sectors[best_sector]
        
        # 根据包裹器圆方程计算目标点
        x_g = rx + l_goal * np.cos(goal_angle)
        y_g = ry + l_goal * np.sin(goal_angle)
        
        # 返回目标点和朝向
        return np.array([[x_g], [y_g], [goal_angle]])
    
    # 示例使用
if __name__ == "__main__":
    # 初始化局部目标包裹器
    wrapper = LocalGoalWrapper(alpha=0.6, beta=0.4)
    
    # 模拟机器人状态 [x, y, theta]
    robot_state = np.array([1.0, 0.5, 0.0])
    
    # 模拟临时目标点（由外部传入）
    temp_goal = np.array([3.0, 1.5])
    
    # 模拟激光雷达数据
    scan_data = {
        'ranges': [5.0] * 360,  # 360度扫描，默认无障碍物
        'angle_min': -np.pi,
        'angle_max': np.pi,
        'range_max': 10.0
    }
    
    # 在某些角度设置障碍物，模拟前方有障碍物
    for i in range(150, 210):
        scan_data['ranges'][i] = 1.0
    
    # 生成局部目标
    local_goal = wrapper.generate_local_goal(robot_state, temp_goal, scan_data)
    
    print(f"临时目标点: {temp_goal}")
    print(f"局部目标点: {local_goal[:2]}")
    print(f"目标朝向: {np.degrees(local_goal[2]):.1f}°")