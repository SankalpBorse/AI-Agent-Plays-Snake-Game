import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from pygame.math import Vector2

# IMPORT YOUR GAME ENVIRONMENT HERE
# Note: You will need to slightly adapt your game file to act as an environment. 
# See the instructions below this code block.
from snake_env import MAIN_AI, cell_number 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 256, 3) # 11 states, 256 hidden nodes, 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake.body[0]
        
        # Points around the head
        point_l = head + Vector2(-1, 0)
        point_r = head + Vector2(1, 0)
        point_u = head + Vector2(0, -1)
        point_d = head + Vector2(0, 1)
        
        # Current directions
        dir_l = game.snake.direction == Vector2(-1, 0)
        dir_r = game.snake.direction == Vector2(1, 0)
        dir_u = game.snake.direction == Vector2(0, -1)
        dir_d = game.snake.direction == Vector2(0, 1)

        # Helper to check collision
        def is_collision(pt):
            if pt.x < 0 or pt.x >= cell_number or pt.y < 0 or pt.y >= cell_number:
                return True
            if pt in game.snake.body[1:]:
                return True
            return False

        state = [
            # Danger straight
            (dir_r and is_collision(point_r)) or 
            (dir_l and is_collision(point_l)) or 
            (dir_u and is_collision(point_u)) or 
            (dir_d and is_collision(point_d)),

            # Danger right
            (dir_u and is_collision(point_r)) or 
            (dir_d and is_collision(point_l)) or 
            (dir_l and is_collision(point_u)) or 
            (dir_r and is_collision(point_d)),

            # Danger left
            (dir_d and is_collision(point_r)) or 
            (dir_u and is_collision(point_l)) or 
            (dir_r and is_collision(point_u)) or 
            (dir_l and is_collision(point_d)),
            
            # Move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location 
            game.fruit.pos.x < game.snake.body[0].x,  # food left
            game.fruit.pos.x > game.snake.body[0].x,  # food right
            game.fruit.pos.y < game.snake.body[0].y,  # food up
            game.fruit.pos.y > game.snake.body[0].y   # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        # Exploration: random moves
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        # Exploitation: network prediction
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

def train():
    record = 0
    agent = Agent()
    game = MAIN_AI()
    
    print("Starting Training...")
    
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory (for the single step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory (experience replay), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')

if __name__ == '__main__':
    train()