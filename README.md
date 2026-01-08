# mujoco_tasksim
Dynamic task simulation of varying NIST assembly tasks utilizing MuJoCo.

1. Set up

Set up a python environment with the python version 3.10.14. Download the packages from the requirements.txt with the specified versions. Please note that the installed version of the package and software from MuJoCo need to be the exact same. If necessary, the mujoco/bin directory has to be added to the path of the python environment.

2. Usage
   
This MuJoCo Simulation is created to simulate structured compliant grippers in different assembly tasks with varying starting positions. This way, the tolerable ranges and failure cases can be observed.

3. Repository Layout
   
The sim.py contains the simulation control. The robot_15.xml and scene_15.xml contain the model for the straight gripper design. The robot_chamf.xml and scene_chamf.xml contain the model for the chamfered gripper design.
The asset directory contains all model assets for the simulation. It needs to be in the same directory as the other files.

Based on the work of Alexander Rother
