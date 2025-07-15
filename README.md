# Giraffe Robot Project 
This repository contains the implementation of the final project for the "Introductio to Robotics" course, focused on the design and control of a "Giraffe" robot to automate microphone handling during Q&A sessions in environments like conference rooms or theaters.

## Robot Description
The robot has 5 Degrees of Freedom (DoF):
- 1 spherical joint at the base: Modeled as 2 revolute joints with intersecting axes.
- 1 prismatic joint: Capable of a long extension to reach different heights.
- 2 revolute joints: To properly orient the microphone, with a specific pitch orientation of 30 degrees with respect to the horizontal.

The robot's primary task is a 4D task (X, Y, Z position and pitch orientation). The system exploits its redundancy (1 DoF) to perform a secondary task, such as minimizing the distance from a desired joint configuration.

## Repository Structure
The repository is structured as a ROS (catkin_ws) workspace.
```
└── giraffe_description
    ├── CMakeLists.txt
    ├── launch
    │   └── display.launch
    ├── package.xml
    ├── rviz
    │   ├── config_frame.rviz
    │   └── config.rviz
    ├── scripts
    │   ├── conf.py
    │   ├── controllers
    │   ├── dynamics.py
    │   ├── kinematics.py
    │   ├── main.py
    │   ├── __pycache__
    │   ├── task_space.py
    │   └── utils
    ├── urdf
    │   ├── generated_urdf
    │   └── giraffe.urdf.xacro
    └── utils
        ├── controllers
        ├── __pycache__
        └── utils
```
- `main.py`: The main entry point for running simulations and tests. It offers an interactive choice between different operating modes (kinematic tests, dynamic simulation, task space simulation, RViz visualization only).

- `conf.py`: Contains global configuration parameters for the robot, initial conditions, controller gains, and trajectory parameters.

- `kinematics.py`: Implements and tests the robot's forward and differential kinematics.

- `dynamics.py`: Contains the logic for the robot's dynamic simulation, including RNEA calculations and forward integration.

- `task_space.py`: Implements task space control, polynomial trajectory generation, and redundancy resolution via null-space projection.

`utils` Folder contains reusable utility functions.

## How to Run the Code
### ROS Environment Setup
Ensure you have a ROS environment (e.g., Noetic) configured.

- Clone the repository
```
git clone https://github.com/MartinaPanini/Giraffe_Robot.git
cd Giraffe_Robot/catkin_ws/src
```

- Compile the workspace:
```
catkin_make
```
- Source the environment:
```
source devel/setup.bash
```
- Run the main node:
```
rosrun giraffe_description main.py
```
You will be prompted to choose the type of test/simulation to run via a text menu.

### Dependencies
ROS (Robot Operating System)

Pinocchio: Robot kinematics and dynamics library.

Numpy: For numerical operations.

Matplotlib: For plot generation.

xacro: For URDF model generation.

rospy, std_msgs, urdf, xacro: ROS packages.

