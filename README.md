# Minimal_TEB

<img src='https://img.shields.io/badge/Python-3.9-blue' alt='bilibili'></a>
<a href="https://github.com/hanruihua/ir-sim"><img src='https://img.shields.io/badge/ir--sim-2.5.0-lightgreen' alt='Paper BibTex'></a>
<a href="https://github.com/casadi"><img src='https://img.shields.io/badge/casadi-3.7.0-red' alt='ROS'></a>

---

This project implements a simplified version of the TEB (Time Elastic Band) algorithm based on the [IR-SIM](https://github.com/hanruihua/ir-sim) platform. This implementation focuses on learning, retaining the core logic of TEB while simplifying the algorithm flow to facilitate understanding and learning.

# Project Structure Overview

- [run.py](#runpy)
- [sim.py](#simpy)
- [TebSolver.py](#tebsolverpy)

## run.py

**Project Entry Script**

| Functionality       | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| Import & Initialize | Import the `SIM_ENV` class and instantiate it with rendering enabled.             |
| Main Loop           | Iterate 3000 times, advancing the simulation via `env.step()` each iteration.     |
| Termination         | Save the animation and exit the loop when the robot reaches the goal or collides. |

---

## sim.py

**Simulation Environment Logic**

| Module            | Responsibility                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| Initialization    | Build the simulation environment on top of `ir-sim.EnvBase`.                                       |
| Global Planning   | Generate a global path from start to goal using the **A\*** algorithm.                             |
| Local Planning    | Call `TebplanSolver` to obtain a locally optimal trajectory.                                       |
| Sensor Processing | Cluster LiDAR data to extract obstacles.                                                           |
| Robot Control     | Compute linear/angular velocities, execute simulation steps, and determine termination conditions. |

---

## TebSolver.py

**TEB (Time Elastic Band) Local Path Planner**

| Dimension   | Content                                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| Modeling    | Formulate a nonlinear programming (NLP) problem with **CasADi**.                                                   |
| Objective   | Path smoothness + time penalty + kinematic constraints.                                                            |
| Constraints | Boundary poses, obstacle-avoidance safety margins, velocity/angular/acceleration limits, etc.                      |
| Solving     | Adaptively adjust the number of trajectory points and return the optimized trajectory along with its time profile. |

# Prerequisite

- `python = 3.9`
- `ir-sim = 2.5.0`
- `casadi = 3.7.0`

# Installation

```bash
git clone https://github.com/whsleep/Minimal_TEB.git
cd Minimal_TEB
pip install -r requirements.txt
```

# Run examples

```shell
python run.py
```

# Demonstration

<img  src="pictures/20obs.gif" width="400" />

# Tutorial

Here is a short Chinese tutorial [Trajectory Optimization](https://www.zhihu.com/column/c_1940366621676905723)

# References

- [RDA Planner](https://github.com/hanruihua/RDA-planner)

  RDA Planner is a high-performance, optimization-based, Model Predictive Control (MPC) motion planner designed for autonomous navigation in complex and cluttered environments. Utilizing the Alternating Direction Method of Multipliers (ADMM), RDA decomposes complex optimization problems into several simple subproblems. This decomposition enables parallel computation of collision avoidance constraints for each obstacle, significantly enhancing computation speed.

- [Intelligent Robot Simulator (IR-SIM)](https://github.com/hanruihua/ir-sim)

  IR-SIM is an open-source, Python-based, lightweight robot simulator designed for navigation, control, and learning. It provides a simple, user-friendly framework with built-in collision detection for modeling robots, sensors, and environments. Ideal for academic and educational use, IR-SIM enables rapid prototyping of robotics and AI algorithms in custom scenarios with minimal coding and hardware requirements.

- [AutoNavRL](https://github.com/harshmahesheka/AutoNavRL)

  This project implements a reinforcement learning-based robot navigation system that enables autonomous navigation in complex environments with obstacles.

- [teb_local_planner](https://github.com/gogongxt/teb_local_planner)

  Transplanted the official teb source code to achieve common optimization and multi-path optimization.
