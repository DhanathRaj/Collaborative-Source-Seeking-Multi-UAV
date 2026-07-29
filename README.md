# Collaborative Source Seeking Using Multiple UAVs

A MATLAB implementation of a **Leader–Follower Cooperative Source Seeking** algorithm for multiple UAVs operating in an unknown scalar field. The framework integrates **gradient-based navigation**, **pseudo-inverse gradient estimation**, **adaptive neural network attitude control**, and **formation control** to enable autonomous cooperative exploration.

---

# Project Overview

Autonomous multi-UAV systems are increasingly used in applications such as:

- Environmental monitoring
- Gas leakage detection
- Radiation source localization
- Search and rescue
- Precision agriculture
- Military reconnaissance

In many practical scenarios, only the leader UAV has direct access to gradient information, while follower UAVs estimate the gradient locally using neighboring measurements. This project presents a cooperative leader–follower architecture where:

- The leader performs gradient ascent toward the source.
- Followers maintain a desired formation.
- Followers estimate the field gradient using a pseudo-inverse least-squares approach.
- Adaptive Neural Networks compensate unknown disturbances in the attitude dynamics.

The complete framework is implemented and validated in MATLAB.

---

# Features

- Multi-UAV Leader–Follower Architecture
- Cooperative Source Seeking
- Gradient-Based Navigation
- Pseudo-Inverse Gradient Estimation
- Adaptive Neural Network Attitude Controller
- Fast Finite-Time Attitude Stabilization
- Formation Control
- Gaussian Source Field Modeling
- Disturbance Estimation
- MATLAB ODE45 Simulation
- Thesis-quality Performance Plots

---

# Control Architecture

```text
                  Gaussian Source Field
                           │
                           ▼
                    Leader UAV
               Gradient-Based Navigation
                           │
                           ▼
          Desired Formation Trajectory
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Follower 1      Follower 2      Follower N
          │                │                │
          └────Pseudo-Inverse Gradient──────┘
                    Estimation
                           │
                           ▼
             Formation Controller
                           │
                           ▼
       Adaptive Neural Network Controller
                           │
                           ▼
                 UAV Dynamics
```

---

# Methodology

The proposed algorithm consists of the following stages.

## 1. Gaussian Scalar Field

A three-dimensional Gaussian scalar field is generated to represent the unknown source.

The leader seeks the maximum field intensity.

---

## 2. Leader Guidance

The leader computes the field gradient and performs gradient ascent.

The translational controller generates the desired thrust vector.

---

## 3. Formation Control

Follower UAVs maintain predefined offsets with respect to the leader.

Formation errors are minimized using nonlinear position and velocity feedback.

---

## 4. Gradient Estimation

Followers do not directly measure the gradient.

Instead, each follower estimates the gradient using neighboring measurements through a regularized pseudo-inverse least-squares method.

---

## 5. Adaptive Neural Network

Unknown disturbances acting on the UAV attitude dynamics are estimated online using a single-hidden-layer adaptive neural network.

The estimated disturbance is compensated within the control law.

---

## 6. Fast Finite-Time Attitude Control

The proposed controller guarantees

- finite-time convergence
- robustness against disturbances
- fast attitude synchronization
- bounded control torque

---

# Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Number of Followers | 4 |
| UAV Mass | 1 kg |
| Gravity | 9.81 m/s² |
| Source Field | Gaussian |
| Numerical Solver | MATLAB ODE45 |
| Attitude Controller | Adaptive Neural Network |
| Formation | Square Formation |

---

# Repository Structure

```text
Collaborative-Source-Seeking-Multi-UAV
│
├── README.md
├── source_seeking_main.m
```

---

# Simulation Results

The MATLAB simulation generates the following outputs.

- 3D UAV trajectories
- Formation tracking performance
- Source seeking convergence
- Gaussian field evolution
- Attitude synchronization error
- Adaptive neural network disturbance estimation
- Control torque history
- Gradient estimation accuracy
- Velocity profiles
- Simulation summary dashboard

---

# MATLAB Requirements

- MATLAB R2022b or later

Required Toolboxes

- Control System Toolbox
- Optimization Toolbox

---

# How to Run

Clone the repository

```bash
git clone https://github.com/DhanathRaj/Collaborative-Source-Seeking-Multi-UAV.git
```

Open MATLAB.

Navigate to the project directory.

Run

```matlab
source_seeking_main
```

The simulation automatically

- initializes the UAVs
- performs cooperative source seeking
- estimates the gradient
- executes adaptive neural network control
- generates all simulation figures

---

# Applications

This work can be extended to

- Swarm Robotics
- Autonomous Drone Fleets
- Environmental Monitoring
- Search and Rescue
- Precision Agriculture
- Hazardous Gas Detection
- Radiation Source Localization
- Autonomous Exploration

---

# Future Work

- ROS2 implementation
- Gazebo simulation
- PX4 SITL integration
- Hardware-in-the-loop simulation
- Real UAV flight experiments
- Obstacle avoidance
- Distributed Kalman Filtering
- Reinforcement Learning based source seeking

---

# Author

**Dhanath Raj CR**

M.Tech Instrumentation and Control Systems

National Institute of Technology Calicut

---

# License

This project is intended for academic and research purposes.

If you use this work in your research, please cite the repository appropriately.

---

## Acknowledgements

This project was developed as part of the M.Tech research work on cooperative autonomous UAV systems, focusing on nonlinear control, adaptive neural networks, formation control, and distributed source-seeking algorithms.
