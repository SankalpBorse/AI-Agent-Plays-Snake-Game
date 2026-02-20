# 🐍 AI Agent Plays Snake Game

This is the Snake Game Built with python pygame library and the Agent is trained using **Deep Q-Learning (DQN)** in PyTorch.
Learned the to code this game using the following youtube video:
https://youtu.be/QFvqStqPCRU

The project contains a training file to train the model from scratch, and a multi-mode playable game where you can watch the AI play, play normally only human, or play against AI Aide by Side

## Description
The model is trained using Reinforcement learrning, Deep Q-Learning (DQN). Instead of viewing the entire visual game board like a human player, the AI agent relies on a simplified, sensory representation of its environment known as a "state." Every fraction of a second, the game feeds the agent an 11-point array of true/false data answering spatial questions: Is there immediate danger ahead, left, or right? Which direction am I currently moving? And where is the food relative to my head? The agent's "brain"—a deep neural network—processes this state and calculates a "Q-value" (Quality score) for three possible actions: keep going straight, turn left, or turn right. The AI simply chooses the action that yields the highest predicted score.

To actually learn these behaviors, the agent relies on a continuous loop of trial, error, and mathematical optimization. When training begins, the AI acts completely randomly (a phase called exploration), frequently crashing into walls but occasionally eating an apple by pure chance. The game environment provides strict feedback: a positive reward (+10) for eating food, and a severe penalty (-10) for dying. By storing these experiences in its memory, the neural network uses the Bellman equation to continuously adjust its internal weights, slowly learning to associate specific states with the actions that maximize future rewards. Over hundreds of simulated games, the AI transitions from random guessing to calculated exploitation, eventually developing the spatial awareness and strategy required to weave around its own tail and survive autonomously.

## Project Structure
```text
├── Font/                   # Font files (PoetsenOne-Regular.ttf)
├── Graphics/               # Snake body, head, tail, and apple PNGs
├── Sound/                  # Game sounds (crunch.wav)
├── model/                  # Trained Model
├── Snake.py                # Main game file (Includes Menu & 3 Modes)
├── snake_env.py            # AI Training Game environment
├── train.py                # Training file contains training loop
├── requirements.txt        
└── README.md               
