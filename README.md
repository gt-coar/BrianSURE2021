# BrianSURE2021
Brian Ko (KyungMin Ko) work for SURE 2021

All of my codes are using pytorch and Wandb. Wandb is great tool for logging the machine learning experiment. You can checkout the detail in the link below. https://docs.wandb.ai/

## REINFORCE
REINFORCE_Batch_Update: Implementation of REINFORCE algorithm and tested with different updating frequency.
REINFORCE_Foward_Backward: It contains my implementation of REINFORCE algorithm with backward method which updates the theta from t= T-1 to t= 1, and tested on both CartPole and LunarLander-v2 envrionment

## REINFORCE with Baseline
Implementation of REINFORCE with baseline with baseline using value function. Have tested with backward, forward, and average loss mehtod on Wandb.

## TRPO
It contains the experiments on comparing TRPO with DQN ACKTR algorithm using Stable Baseline 3 implementation on Lunar Lander-v2

## TD Actor Critic
Implementation of TD Actor Critic with sperate neural network, worked on comparing different optimizers with Adam and RMSprop.


