import os
import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
from agent_pytorch import DeepQLearningAgent, PolicyGradientAgent, AdvantageActorCriticAgent
import json

# Set device to GPU if available (useful for running on Colab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Version and global variables
version = 'v17.1'

# Get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # Keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# Define the number of episodes and logging frequency
episodes = 10000
log_frequency = 500
games_eval = 8

# Define the DQN model
class DQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNModel, self).__init__()
        # input_shape[0] = Antall kanaler (fra frames)
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Dynamisk beregning av output-størrelse etter konvolusjonslag
        conv_output_size = self._get_conv_output_size(input_shape)

        # Fullt tilkoblede lag
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)  # Dummy input
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten using reshape
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                           buffer_size=buffer_size, version=version)
agent.device = device  # Manually assign the device since DeepQLearningAgent doesn't take it as an argument

# Check agent type
if isinstance(agent, DeepQLearningAgent):
    agent_type = 'DeepQLearningAgent'
elif isinstance(agent, PolicyGradientAgent):
    agent_type = 'PolicyGradientAgent'
elif isinstance(agent, AdvantageActorCriticAgent):
    agent_type = 'AdvantageActorCriticAgent'
print('Agent is {:s}'.format(agent_type))

# Setup epsilon range and decay rate for epsilon, reward type, and update frequency
if agent_type == 'DeepQLearningAgent':
    epsilon, epsilon_end = 1.0, 0.01
    reward_type = 'current'
    sample_actions = False
    n_games_training = 8 * 16
    decay = 0.97
    if supervised:
        # Lower the epsilon since some starting policy has already been trained
        epsilon = 0.01
        # Load the existing model from a supervised method
        agent.load_model(file_path='models/{:s}'.format(version))

elif agent_type == 'PolicyGradientAgent':
    epsilon, epsilon_end = -1, -1
    reward_type = 'discounted_future'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 16
    decay = 1

elif agent_type == 'AdvantageActorCriticAgent':
    epsilon, epsilon_end = -1, -1
    reward_type = 'current'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 32
    decay = 1

# Use only for DeepQLearningAgent if the buffer should be filled
if agent_type == 'DeepQLearningAgent':
    if supervised:
        try:
            agent.load_buffer(file_path='models/{:s}'.format(version), iteration=1)
        except FileNotFoundError:
            pass
    else:
        games = 512
        env = SnakeNumpy(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
                         games=games, frame_mode=True, obstacles=obstacles, version=version)
        ct = time.time()
        _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                       epsilon=epsilon, verbose=True, reset_seed=False,
                       frame_mode=True, total_frames=games * 64)
        print('Playing {:d} frames took {:.2f}s'.format(games * 64, time.time() - ct))

# Create environments
env = SnakeNumpy(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
                 games=n_games_training, frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
                  games=games_eval, frame_mode=True, obstacles=obstacles, version=version)

# Training loop
model_logs = {'iteration': [], 'reward_mean': [],
              'length_mean': [], 'games': [], 'loss': []}

for index in tqdm(range(episodes)):
    if agent_type == 'DeepQLearningAgent':
        _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon, n_games=n_games_training,
                             record=True, sample_actions=sample_actions, reward_type=reward_type,
                             frame_mode=True, total_frames=n_games_training, stateful=True)
        loss = agent.train_agent(batch_size=64, num_games=n_games_training, reward_clip=True)

    elif agent_type == 'AdvantageActorCriticAgent':
        _, _, total_games = play_game2(env, agent, n_actions, epsilon=epsilon,
                                       n_games=n_games_training, record=True,
                                       sample_actions=sample_actions, reward_type=reward_type,
                                       frame_mode=True, total_games=n_games_training * 2)
        loss = agent.train_agent(batch_size=agent.get_buffer_size(), num_games=total_games, reward_clip=True)

    if agent_type in ['PolicyGradientAgent', 'AdvantageActorCriticAgent']:
        # For policy gradient algorithm, we only take current episodes for training
        agent.reset_buffer()

    # Check performance every once in a while
    if (index + 1) % log_frequency == 0:
        current_rewards, current_lengths, current_games = play_game2(env2, agent, n_actions, n_games=games_eval,
                                                                     epsilon=-1, record=False, sample_actions=False,
                                                                     frame_mode=True, total_frames=-1,
                                                                     total_games=games_eval)
        model_logs['iteration'].append(index + 1)
        model_logs['reward_mean'].append(round(int(current_rewards) / current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths) / current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']].to_csv(
            'model_logs/{:s}.csv'.format(version), index=False)

    # Copy weights to target network and save models
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index + 1))
        epsilon = max(epsilon * decay, epsilon_end)
