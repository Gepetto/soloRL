# soloRL

Environment code for the paper "Controlling the Solo12 Robot with Deep Reinforcement Learning"  
Preprint: https://hal.laas.fr/hal-03761331
Video: https://youtu.be/t-67qBxNyZI

The environment contains the state space, action space and reward function used to setup the learning of locomotion skills for the solo12 quadruped. 
It contains details about the curriculum used and domain/dynamic randomization strategies used in this work.

You can use your favorite RL algorithm to train with this environment that is based on the raisim simulator www.raisim.com. 
Install [raisim](www.raisim.com) and [raisimGymTorch](https://raisim.com/sections/RaisimGymTorch.html), plug the soloRL repo in raisimGymTorch/raisimGymTorch/env/envs directory. Re-build and train. 

Coming soon example RL training/testing based on stable-baselines + example gym environment based on Pybullet in python.

NOTE: If you have a Solo12 and wish to try our policies, we also provide the policy checkpoints used and the interface code with the robot in https://gitlab.laas.fr/paleziart/quadruped-rl 

This neural network runs on the robot with a speed of 10 μs!

File description: <br />
Environment.hpp: Enviornment description in format suitable for the raisim simulator. <br />
cfg.yaml: main training parameters. <br />
cfg_terrain.yaml: training parameters with non-flat terrains (check the papers for details). <br />
NoiseSampler.h: noise vector objects  (in eigen) for domain randomization. <br />
HeightTerrain.hpp: heightmap description used in training non-flat terrains.
