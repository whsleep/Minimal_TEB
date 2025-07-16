import casadi as ca
import numpy as np

class TebplanSolver:
    def __init__(self,
                 start: np.ndarray,
                 end: np.ndarray,
                 obs: np.ndarray,
                 n = None,          # 允许 None -> 自动设定轨迹点数量
                 vmax: float = 1.5,
                 wmax: float = 1.5,
                 vamax: float = 1.5,
                 rmin: float = 0.5,
                 safe_dis: float = 1.2,
                 auto_hz: float = 10):        # 控制频率
        self.start = np.copy(np.asarray(start, float))  
        self.end = np.copy(np.asarray(end, float))    
        self.obs      = np.copy(np.asarray(obs, float))
        self.vmax     = vmax
        self.wmax     = wmax
        self.vamax    = vamax
        self.rmin     = rmin
        self.safe_dis = safe_dis
        self.auto_hz  = auto_hz

        # 1. 自动计算 n
        if n is None:
            self._auto_size()
        else:
            self.n = n

        self._last_sol = None

    def _auto_size(self):
        dist = np.linalg.norm(self.end[:2] - self.start[:2])
        t_min = dist / self.vmax
        self.n = max(5, int(np.ceil(t_min * self.auto_hz)))
        print(f'[Auto] dist={dist:.2f}, t_min={t_min:.2f}, n={self.n}')

    # ---------- 内部 ----------
    def _build_nlp(self, obs_now):
        n = self.n
        x  = ca.SX.sym('x', n)
        y  = ca.SX.sym('y', n)
        θ  = ca.SX.sym('θ', n)
        ΔT = ca.SX.sym('ΔT', n+1)
        w  = ca.vertcat(x, y, θ, ΔT)

        pts = [ca.DM(self.start)]
        for k in range(n):
            pts.append(ca.vertcat(x[k], y[k], θ[k]))
        pts.append(ca.DM(self.end))

        def norm_angle(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        res = []
        res_va  = []

        for i in range(n+1):
            p0, p1   = pts[i][:2], pts[i+1][:2]
            th0, th1 = pts[i][2], pts[i+1][2]
            seg = ca.norm_2(p1 - p0)
            dt  = ΔT[i]

            res.append(seg)
            res.append(dt**2)
            if 1 <= i <= n:
                dists = [ca.norm_2(pts[i][:2] - o.T) for o in obs_now]
                dmin  = ca.mmin(ca.vertcat(*dists))
                res.append(ca.fmax(0, self.safe_dis - dmin))
            if i <= n-1:
                # 计算 v_i, v_{i+1}, ω_i, ω_{i+1}
                seg_i   = ca.norm_2(pts[i+1][:2] - pts[i][:2])
                seg_ip1 = ca.norm_2(pts[i+2][:2] - pts[i+1][:2])
                dt_i   = ΔT[i]
                dt_ip1 = ΔT[i+1]
                v_i   = seg_i   / (dt_i   + 1e-6)
                v_ip1 = seg_ip1 / (dt_ip1 + 1e-6)

                # 加速度
                denom = 0.5 * (dt_i + dt_ip1) + 1e-6
                va  = (v_ip1 - v_i) / denom

                res_va.append(ca.fmax(0, ca.fabs(va) - self.vamax))
            
            v = seg / (dt + 1e-6)
            res.append(ca.fmax(0, ca.fabs(v) - self.vmax))
            dth = norm_angle(th1 - th0)
            ω   = dth / (dt + 1e-6)
            res.append(ca.fmax(0, ca.fabs(ω) - self.wmax))
            l0 = ca.vertcat(ca.cos(th0), ca.sin(th0))
            l1 = ca.vertcat(ca.cos(th1), ca.sin(th1))
            d  = p1 - p0
            cross = (l0[0]+l1[0])*d[1] - (l0[1]+l1[1])*d[0]
            res.append(10*cross)
            # 计算转弯半径（考虑速度方向）
            r = v / (ca.fabs(ω) + 1e-6)
            # 确保转弯半径的绝对值大于等于最小转弯半径
            res.append(10*ca.fmax(0, ca.fabs(r) - self.rmin))


        residuals = ca.vertcat(*res,*res_va)
        # residuals = ca.vertcat(*res)
        nlp = {'x': w, 'f': 0.5 * ca.dot(residuals, residuals)}
        opts = {'ipopt.print_level': 0, 'print_time':0,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.max_iter': 500}
        return ca.nlpsol('solver', 'ipopt', nlp, opts)

    def _init_guess(self):
        if self._last_sol is not None:
            return self._last_sol
        n = self.n
        return np.hstack([
            np.linspace(self.start[0], self.end[0], n+2)[1:-1],
            np.linspace(self.start[1], self.end[1], n+2)[1:-1],
            np.unwrap(np.linspace(self.start[2], self.end[2], n+2))[1:-1],
            np.full(n+1, 0.2)
        ])

    # ---------- 公开 ----------
    def solve(self,
            obs_new=None,
            start_new=None,
            end_new=None,
            pos_th: float = 0.05,   # 位置变化阈值 [m]
            ang_th: float = 0.10):  # 角度变化阈值 [rad]
        """
        在线重规划入口
        obs_new    : 新障碍物坐标 (Nx2)
        start_new  : 新起点 (3,) 或 None
        end_new    : 新终点 (3,) 或 None
        pos_th     : 位置变化触发阈值
        ang_th     : 角度变化触发阈值
        返回:
        traj, cost
        """

        # 1. 障碍物更新
        if obs_new is not None:
            self.obs = np.asarray(obs_new, float)

        # 2. 起点 / 终点更新 + 偏差检测
        dirty_n = False
        if start_new is not None:
            start_new = np.asarray(start_new, float)
            if (np.linalg.norm(start_new[:2] - self.start[:2]) > pos_th or
                    abs(np.arctan2(np.sin(start_new[2] - self.start[2]),
                                np.cos(start_new[2] - self.start[2]))) > ang_th):
                self.start = np.copy(start_new) 
                dirty_n = True

        if end_new is not None:
            end_new = np.asarray(end_new, float)
            if (np.linalg.norm(end_new[:2] - self.end[:2]) > pos_th or
                    abs(np.arctan2(np.sin(end_new[2] - self.end[2]),
                                np.cos(end_new[2] - self.end[2]))) > ang_th):
                self.end = np.copy(end_new) 
                dirty_n = True

        # 3. 只有起点/终点变化时才重算 n 并清空上一次解
        if dirty_n:
            self._auto_size()
            self._last_sol = None          # 强制重新生成初值

        # 4. 其余流程保持不变
        solver = self._build_nlp(self.obs)
        sol = solver(x0=self._init_guess(), lbx=-10, ubx=10)
        if sol['f'] is None:
            return None, np.inf
        w_opt = np.array(sol['x']).flatten()
        self._last_sol = w_opt
        traj = np.vstack([self.start,
                        np.column_stack([w_opt[:self.n],
                                        w_opt[self.n:2*self.n],
                                        w_opt[2*self.n:3*self.n]]),
                        self.end])
        # 提取时间差序列（最后 n+1 个元素）
        dT = w_opt[3*self.n:]  # 时间差序列 ΔT
        
        return traj, dT  # 返回轨迹和时间差序列

