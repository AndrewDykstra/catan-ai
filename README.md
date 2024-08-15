# Catan AI Agent - CS 4100
### Group Members: Andrew Dykstra, Anne Crinnion, and Charles Zhao

The current version V1.1.1 has a game environment developed. There's more description below of what's implemented so far and what still needs to be added.

## Libraries Needed
pip instal the following libraries
- gym
- numpy
- pytorch
- matplotlib

## Current Features
- [x] Game Environment: Currently, there is a basic initialization of the game board, resources, and player states.
- [x] Actions: There are basic actions including building roads and settlements but we need to add more.
- [x] More Game Mechanics/Moves: We need to develop more actions that the AI can take within the game of Catan.
- [X] Reinforcement Learning: A Q-learning agent with a neural network to learn optimal strategies through self-play. Need to grow beyond initial training still
- [x] Opponent AI: Decide on how we want our opponents to interact with the game. They could be clones or they could have set rules.
- [x] Adjacency: Logic to compute which tiles are adjacent and which tiles a settlement will have access to.
- [x] Training: The agent interacts with the environment, takes actions, receives rewards, and updates its policy based on the feedback.
- [x] Turns: The agent needs to take turns while iterating to ensure that it doesn't get stuck in an infinite loop.
- [x] Evaluation: Ideally implement a way to track how our AI Agent is performing in the game.

## What needs to be added still
- [ ] Resource Management: Resource management is very simplistic so we need to implement this based on our decision tree.

## Feedback from Check-in + Discussion
### What to reward the Agent for:
- [x] Earning a Victory Point
- [x] Winning the game

### What to Penalize the Agent for
- [x] Attempting to make moves that aren't possible/doesn't have the resources for
- [x] Every turn -1 (Modified to be -3)

### Feedback
Train against a bot with set rules then once it's compotent against other versions of itself   
Actions to add: Trading with bank, trade 4 of same resource for 1 of your choice, then ports could be 3-1 and specialized is 2-1 for a port   
Future: Add robber, specialized cards   