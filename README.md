# Catan AI Agent - CS 4100
### Group Members: Andrew Dykstra, Anne Crinnion, and Charles Zhao

The current version V1.0.2 has a game environment developed. There's more description below of what's implemented so far and what still needs to be added.

## Libraries Needed
pip instal the following libraries
- gym
- numpy
- tensorflow

## Current Features
- [x] Game Environment: Currently, there is a basic initialization of the game board, resources, and player states.
- [x] Actions: There are basic actions including building roads and settlements but we need to add more.
- [ ] Reinforcement Learning: A Q-learning agent with a neural network to learn optimal strategies through self-play. This is partway complete

## What needs to be added still
- [ ] Training: The agent interacts with the environment, takes actions, receives rewards, and updates its policy based on the feedback.
- [ ] More Game Mechanics/Moves: We need to develop more actions that the AI can take within the game of Catan.
- [ ] Resource Management: Resource management is very simplistic so we need to implement this based on our decision tree.
- [ ] Opponent AI: Decide on how we want our opponents to interact with the game. They could be clones or they could have set rules.
- [ ] Evaluation: Ideally implement a way to track how our AI Agent is performing in the game.