import casadi as ca
import numpy as np


class TrajOpt:
    def __init__(self,
                 dt_seg=0.2,          # 每段最大时间步长（秒）
                 vmax=1.0, wmax=1.5, rmin=0.5, safe_dis=0.9,
                 obstacle_penalty=1.0,
                 curvature_penalty=10.0):
        """
        dt_seg           用于自动计算 n 的“时间分辨率”
        vmax, wmax, rmin 速度/角速度/最小转弯半径
        safe_dis         与障碍物最小安全距离
        obstacle_penalty 安全距离软约束权重
        curvature_penalty 非完整/曲率约束权重
        """
        self.dt_seg       = dt_seg
        self.vmax         = vmax
        self.wmax         = wmax
        self.rmin         = rmin
        self.safe_dis     = safe_dis
        self.w_obs        = obstacle_penalty
        self.w_curv       = curvature_penalty

        # 决策变量占位，将在 solve 中动态创建
        self.x  = None
        self.y  = None
        self.th = None
        self.dt = None
        self.w_sym = None

    # ---------- 工具 ----------
    @staticmethod
    def _norm_angle(a):
        return ca.atan2(ca.sin(a), ca.cos(a))

    # ---------- 构造 & 求解 ----------
    def solve(self, start, end, obs=None, x0=None):
        """
        start/end: [x, y, theta]
        obs: list/array [[x1,y1], ...]
        其余与 TrajOpt 原始接口一致
        """
        start = np.asarray(start, float).ravel()
        end   = np.asarray(end,   float).ravel()
        obs   = np.asarray(obs) if obs is not None else np.empty((0, 2))

        # 1. 自动计算 n
        dist = np.linalg.norm(end[:2] - start[:2])
        tmin = dist / self.vmax               # 直线匀速所需最短时间
        n = max(2, int(np.ceil(tmin / self.dt_seg)))

        # 2. 根据 n 重新生成符号变量
        self.x  = ca.SX.sym('x', n)
        self.y  = ca.SX.sym('y', n)
        self.th = ca.SX.sym('th', n)
        self.dt = ca.SX.sym('dt', n + 1)
        self.w_sym = ca.vertcat(self.x, self.y, self.th, self.dt)

        # 3. 构造轨迹点
        pts = [start] + [np.array([self.x[k], self.y[k], self.th[k]])
                         for k in range(n)] + [end]

        # 4. 残差构建（完全沿用原逻辑）
        res = []
        for i in range(n + 1):
            p0, p1   = pts[i][:2], pts[i + 1][:2]
            th0, th1 = pts[i][2], pts[i + 1][2]
            seg = ca.norm_2(p1 - p0)
            dt  = self.dt[i]

            # 长度 + 时间
            res.append(seg)
            res.append(dt)

            # 安全距离
            if 1 <= i <= n and obs.shape[0]:
                dists = [ca.norm_2(pts[i][:2] - o) for o in obs]
                dmin  = ca.mmin(ca.vertcat(*dists))
                res.append(self.w_obs * ca.fmax(0, self.safe_dis - dmin))

            # 速度 / 角速度
            v   = seg / (dt + 1e-6)
            dth = self._norm_angle(th1 - th0)
            w   = dth / (dt + 1e-6)
            res.append(ca.fmax(0, ca.fabs(v) - self.vmax))
            res.append(ca.fmax(0, ca.fabs(w) - self.wmax))

            # 非完整运动学
            l0 = ca.vertcat(ca.cos(th0), ca.sin(th0))
            l1 = ca.vertcat(ca.cos(th1), ca.sin(th1))
            d  = p1 - p0
            cross = (l0[0] + l1[0]) * d[1] - (l0[1] + l1[1]) * d[0]
            res.append(self.w_curv * cross)

            # 最小转弯半径
            r = v / (ca.fabs(w) + 1e-6)
            res.append(self.w_curv * ca.fmax(0, r - self.rmin))

        residuals = ca.vertcat(*res)
        nlp = {'x': self.w_sym,
               'f': 0.5 * ca.dot(residuals, residuals)}
        solver = ca.nlpsol('solver', 'ipopt', nlp,
                           {'ipopt.print_level': 0, 'print_time': 0})

        # 5. 初始猜测
        if x0 is None:
            x0 = np.hstack([
                np.linspace(start[0], end[0], n + 2)[1:-1],
                np.linspace(start[1], end[1], n + 2)[1:-1],
                np.unwrap(np.linspace(start[2], end[2], n + 2))[1:-1],
                np.full(n + 1, 0.2)
            ])

        res = solver(x0=x0, lbx=-10, ubx=10)
        w_opt = np.array(res['x']).flatten()

        traj = np.vstack([start,
                          np.column_stack([w_opt[:n],
                                           w_opt[n:2 * n],
                                           w_opt[2 * n:3 * n]]),
                          end])
        dt_seg = w_opt[3 * n:]
        return traj, dt_seg


# ------------------ 使用示例 ------------------
if __name__ == '__main__':
    opt = TrajOptAuto(dt_seg=0.3)  # 每段 0.3 s
    traj, dt_seg = opt.solve(start=[0, 0, 0],
                             end=[5, 3, 1.57],
                             obs=[[2, 1.5]])
    print('自动计算 n =', len(traj) - 2)  # 打印实际中间点个数
    print('轨迹形状:', traj.shape)         # (n+2, 3)