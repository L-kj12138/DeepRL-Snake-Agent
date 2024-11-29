import torch
import numpy as np
from game_environment import SnakeNumpy
from agent_pytorch import DeepQLearningAgent
import argparse

# Argument parser for command-line arguments or Colab compatibility
parser = argparse.ArgumentParser(description="Evaluate the trained Snake RL agent")
parser.add_argument("--model_path", type=str, default="models/v17.1/model_180000.pth", help="Path to the trained model (.pth file)")
parser.add_argument("--board_size", type=int, default=10, help="Size of the Snake board")
parser.add_argument("--frames", type=int, default=2, help="Number of frames used as input")
parser.add_argument("--games", type=int, default=10, help="Number of games to evaluate")
parser.add_argument("--max_time_limit", type=int, default=300, help="Maximum time steps per game")
parser.add_argument("--obstacles", action="store_true", help="Enable obstacles in the environment")
args = parser.parse_args([])  # Empty list for Colab compatibility

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent(model_path, board_size, frames, games, max_time_limit, obstacles):
    """
    Evaluates the trained agent in the Snake environment.
    Returns a dictionary with average reward and game length.
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=4, buffer_size=10000)
    agent.dqn_model.load_state_dict(torch.load(model_path, map_location=device))
    agent.device = device
    agent.dqn_model.to(device)
    print("Model loaded successfully!")

    # Setup the environment
    env = SnakeNumpy(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
                     games=games, obstacles=obstacles)
    
    total_rewards = []
    total_lengths = []

    # Play games and record performance
    print(f"Evaluating the agent over {games} games...")
    for game_idx in range(games):
        state = env.reset()
        done = [False] * games
        rewards = np.zeros(games, dtype=np.float32)
        steps = 0

        while not all(done):
            legal_moves = env.get_legal_moves()
            actions = agent.move(state, legal_moves)
            # Handle additional return values from env.step()
            next_state, reward, done, *_ = env.step(actions)
            rewards += reward
            state = next_state
            steps += 1

        total_rewards.append(np.mean(rewards))
        total_lengths.append(steps)

        print(f"Game {game_idx + 1}: Reward={np.mean(rewards):.2f}, Length={steps}")

    # Calculate average performance
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    print("\n--- Evaluation Results ---")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Game Length: {avg_length:.2f}")
    print("--------------------------")
    
    # Return results as a dictionary
    return {
        "average_reward": avg_reward,
        "average_game_length": avg_length,
        "individual_rewards": total_rewards,
        "individual_lengths": total_lengths
    }

if __name__ == "__main__":
    # Evaluate agent and get results
    results = evaluate_agent(args.model_path, args.board_size, args.frames, args.games, args.max_time_limit, args.obstacles)
    print("\nReturned Results:")
    print(results)
