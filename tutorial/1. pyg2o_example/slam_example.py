import numpy as np
import g2o

import matplotlib.pyplot as plt
 
 
gt = {  '0': {"state_vector": [0,0,0], "connected_frame":['world', '1']},
            '1': {"state_vector": [1,0,np.pi/2], "connected_frame":['0', '2']},
            '2': {"state_vector": [1,1,np.pi], "connected_frame":['1', '3']},
            '3': {"state_vector": [0,1,-np.pi/2], "connected_frame":['2', '4']},
            '4': {"state_vector": [0,0,0], "connected_frame":['3', '0']}    }
 
 
odometry = [
            [0, 0, 0],
            [1.0404914245326837, 0.032285132702715345, 1.548058646294197],
            [1.007675476476252, 0.8930941070656758, 3.0668693044078004],
            [-0.008188478039399927, 0.9558269451002708, 4.646574416535024],
            [-0.03589007133357165, 0.020891631753120143, 6.274558744445183],
            ]
 
 
 
 
def quat_mult(q1, q2):
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])
 
def quat_inv(q):
    """Quaternion inverse."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
 
def quat_to_rot(q):
    """Quaternion to rotation matrix."""
    w, x, y, z = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                  [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                  [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
    return R
 
def quat_diff(q1, q2):
    """Quaternion difference."""
    return quat_mult(q2, quat_inv(q1))
 
 
 
 
 
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        # solver = g2o.BlockSolverX(g2o.LinearSolverCholmodX())
        solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
 
        # solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(True)
 
    def optimize(self, max_iterations=20):
        print('num vertices:', len(super().vertices()))
        print('num edges:', len(super().edges()), end='\n\n')
        super().initialize_optimization()
        super().optimize(max_iterations)
        super().save("out.g2o")
 
 
    def add_vertex3(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
 
    def add_edge3(self, vertices, measurement,
                      information=np.identity(6),
                      robust_kernel=None):
 
        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)
 
        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)
 
    def get_pose3(self, id):
        return self.vertex(id).estimate()
 
    def add_vertex2(self, id, pose, fixed=False):
        v_se2 = g2o.VertexSE2()
        v_se2.set_id(id)
        # v_se2.set_estimate(pose)
        v_se2.set_estimate_data(pose)
        v_se2.set_fixed(fixed)
        super().add_vertex(v_se2)
 
    def add_edge2(self, vertices, measurement,
                 information=np.identity(3),
                 robust_kernel=None):
 
        edge = g2o.EdgeSE2()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)
 
        # edge.set_measurement(measurement)  # relative pose
        edge.set_measurement(g2o.SE2(measurement[0], measurement[1], measurement[2]))
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)
 
    def add_edge_from_state2(self, vertices,
                            information=np.identity(3),
                            robust_kernel=None):
 
        edge = g2o.EdgeSE2()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)
 
        edge.set_measurement_from_state()  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)
 
    def get_pose2(self, id):
        return self.vertex(id).estimate()
 
 
 
 
PGO = PoseGraphOptimization()
 
 
# 顶点节点就是每个节点的总的里程计的值
PGO.add_vertex2(0, [0, 0, 0], fixed=True)
PGO.add_vertex2(1, [1.0404914245326837, 0.032285132702715345, 1.548058646294197], fixed=True)
PGO.add_vertex2(2, [1.007675476476252, 0.8930941070656758, 3.0668693044078004])
PGO.add_vertex2(3, [-0.008188478039399927, 0.9558269451002708, 4.646574416535024])
PGO.add_vertex2(4, [-0.03589007133357165, 0.020891631753120143, 6.274558744445183])
 
# 边就是每次单步走的里程，或者是回环约束； 两个节点之间的距离； todo 第2，3个参数是什么意思 是两帧之间的相对位姿
PGO.add_edge2([0, 1], [1.0404914245326837, 0.032285132702715345, 1.548058646294197], information=20 * np.identity(3))
PGO.add_edge2([1, 2], [0.8598403696701142, 0.05237857840964698, 1.5188106581136034], information=20 * np.identity(3))
PGO.add_edge2([2, 3], [1.0177124423228865, 0.01328035365638311, 1.579705112127223], information=20 * np.identity(3))
PGO.add_edge2([3, 4], [0.9347330338011522, 0.033846328241988986, 1.6279843279101593], information=20 * np.identity(3))
PGO.add_edge2([4, 0], [-0.005390100686657372, 0.0075717125203589725, 0.014532977834258803], information=100 * np.identity(3))
PGO.add_edge2([4, 2], [0.972138664497324, 1.0133949927263823, 3.1359965621109054], information=100 * np.identity(3))
 
PGO.optimize()
 
oposes = []
for i in range(5):
    temp0 = PGO.get_pose2(i)
    temp = PGO.get_pose2(i).vector()
    oposes.append(PGO.get_pose2(i).vector())
# oposes = np.array(oposes)
 
print(oposes)
 
 
 
 
 
 
 
# plot the trajectory
for frame in gt:
    ref_pose = gt[frame]["state_vector"]
    next_frame = gt[frame]["connected_frame"][1]
    next_pose = gt[next_frame]["state_vector"]
    plt.plot([ref_pose[0], next_pose[0]], [ref_pose[1], next_pose[1]], 'r-o', label='gt')
 
for i, ref_pose in enumerate(oposes):
    if i + 1 >= len(oposes):
        break
    next_pose = oposes[i + 1]
    plt.plot([ref_pose[0], next_pose[0]], [ref_pose[1], next_pose[1]], 'b-o', label='op-aft')
 
 
for i,ref_pose in enumerate(odometry):
    if i+1 >= len(odometry):
        break
    next_pose = odometry[i+1]
    plt.plot([ref_pose[0], next_pose[0]], [ref_pose[1], next_pose[1]], 'g-o', label='odometry')
 
 
plt.show()