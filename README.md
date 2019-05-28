# udacity-p2_continuous_control
This is a repository for project 2-continuous-control of udacity DRLND program

Project Details
This project is one of the udactiy deep reinforcement learning nano degree program. The environment you will work with is Unity ML-Agents Reacher. please refer to details [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) for the enviroment.

The goal of this project is to train an agent to move a double-jointed arm to a target location and maintain its position at the target location for as many time steps as possible.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Distributed Training
For this project, multi-agents version of the Unity enviroment which contains 20 identical agents, each with its own copy of the environments has been used. This version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

Solve the Environments.
Agents must get an average score of +30 (over 100 consecutive episodes, and over all agents)

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

