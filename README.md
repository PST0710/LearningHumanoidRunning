# LearningHumanoidRunning

![climb_down](https://github.com/user-attachments/assets/0cbca7ab-ade9-4e77-9b5c-9d2fafead47f)



This project is modified from https://github.com/rohanpsingh/LearningHumanoidWalking, and on the basis of HumanoidWalking, further provides a running pose integrated with hand movements.

## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
LearningHumanoidWalking/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
├── trained/             <-- Contains pretrained model for JVRC
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.7.11  
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==2.2.0
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - ray==1.9.2
  - transforms3d
  - matplotlib
  - scipy

## Usage:

Environment names supported:  

| Task Description                | Environment name |
|---------------------------------|------------------|
| Basic Walking Task              | 'jvrc_walk'      |
| Stepping Task (using footsteps) | 'jvrc_step'      |
| Walking Task (using arm)        | 'jvrc_arm'       |
| run Task (only using leg)       | 'jvrc_run'       |
| run Task (using leg and arm)    | 'jvrc_run_arm'       |


#### **To train:** 

```
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  


#### **To play:**

We need to write a script specific to each environment.    
For example, `debug_stepper.py` can be used with the `jvrc_step` environment.  
```
$ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>
```

#### **What you should see:**





https://github.com/user-attachments/assets/08628f41-29f4-463e-947a-f9cd4d0b210c

