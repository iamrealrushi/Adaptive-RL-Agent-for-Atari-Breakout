# Adaptive RL Agent for Atari Breakout 
This project implements a Deep Q-Network (DQN) agent that plays Atari Breakout while adapting to dynamically changing game conditions. It tackles a modified environment where difficulty shifts unpredictably — testing the agent's robustness, adaptability, and learning capacity.

## Required packages
The code will work on versions of python that are below 3.13.

1. Tensorflow
2. gymnasium
3. ale-py
4. opencv-python

## Project structure

| File                              | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `main.py`                         | Training script for the DQN agent                    |
| `evaluation.py`                   | Evaluates a trained model and visualizes performance |
| `dqn_breakout_weights.weights.h5` | Saved weights after training                         |
| `README.md`                       | Project overview and usage instructions              |
| `report.txt`                      | List of Python dependencies                          |

