# **Snake Reinforcement Learning**

This project implements a Deep Reinforcement Learning (DRL) agent to play the Snake game using PyTorch. The agent uses neural networks to predict the best actions and is trained with various strategies, including Deep Q-Learning, Policy Gradient, and Advantage Actor-Critic.

---

## **Features**
- Train an agent to play Snake on a customizable grid size.
- Support for multiple reinforcement learning algorithms:
  - **Deep Q-Learning (DQN)**
  - **Policy Gradient**
  - **Advantage Actor-Critic (A2C)**
- Replay Buffer for experience replay.
- Supports both training and evaluation modes.
- Designed to run on GPU for faster training.

---

## **Project Structure**

| **File/Directory**        | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| `training.py`             | Main script for training the DRL agent.                                        |
| `agent_pytorch.py`        | Contains implementations of DQN, Policy Gradient, and A2C agents.             |
| `game_environment.py`     | Environment setup for the Snake game.                                          |
| `game_visualization.py`   | Converts gameplay logs to video (MP4 format).                                  |
| `replay_buffer.py`        | Implements the Replay Buffer for experience storage.                           |
| `model_config/`           | Configuration files (JSON format) for training setups.                        |
| `models/`                 | Directory to save trained models.                                              |
| `model_logs/`             | Directory to store training logs (e.g., CSV files).                           |
| `requirements.txt`        | Python dependencies for the project.                                           |

---

## **Getting Started**

### **1. Clone the repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/DeepRL-Snake-Agent.git
cd DeepRL-Snake-Agent
```

### **2. Set up Python environment**
It is recommended to use a virtual environment or Conda to manage dependencies:
```bash
conda create -n snake_rl python=3.8
conda activate snake_rl
pip install -r requirements.txt
```

---

## **Training the Agent**

To train an agent, run the following command:
```bash
python training.py
```

### **Optional Arguments**
- Modify the training configurations in `model_config/v17.1.json`.
- You can specify:
  - `episodes`: Number of training episodes.
  - `log_frequency`: How often logs and models are saved.
  - `reward_type`: Select "current" or "discounted_future" for reward strategies.

---

## **Pre-Trained Models**

Download pre-trained models here:
- [Pre-Trained Model v17.1](https://drive.google.com/link-to-model)

Place the downloaded model in the `models/` directory and use it for evaluation:
```bash
python evaluate.py --model_path models/v17.1.pth
```

---

## **Results and Logs**

- Training logs are saved in the `model_logs/` directory as CSV files.
- Models are automatically saved to the `models/` directory every few episodes.

---

## **Dependencies**

This project requires the following libraries:
- `numpy`
- `pandas`
- `torch`
- `torchvision`
- `matplotlib`
- `tqdm`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## **How to Use with Google Colab**

To run this project on Google Colab:
1. Upload the project folder to Google Drive.
2. Mount Google Drive in Colab and navigate to the project folder:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/DeepRL-Snake-Agent
   ```
3. Install dependencies and run the training or evaluation scripts.

---

## **License**

This project is for educational purposes and open to use under [MIT License](https://opensource.org/licenses/MIT).

---

## **Acknowledgements**

- This project was built as part of a reinforcement learning assignment.
- Thanks to PyTorch and OpenAI Gym for tools and inspiration.

## Evaluation Results
**Average Reward:** -0.41
**Average Game Length:** 301.0
**Individual Rewards:** -0.5, -0.2, -0.6, -0.3, -0.6, -0.4, -0.6, -0.4, -0.2, -0.4
**Individual Game Lengths:** 301, 301, 301, 301, 301, 301, 301, 301, 301, 301

## Gameplay Videos - Note

The gameplay videos generated during the evaluation demonstrate the current performance of the agent. 
However, the agent's behavior is not optimal and does not perform as expected in certain scenarios. 

Further adjustments to the training process, hyperparameters, or reward structure may improve the agent's 
performance in future iterations.

Videos:
1. [game_visual_v17.1_180000_14_ob_0.mp4](images/game_visual_v17.1_180000_14_ob_0.mp4)
2. [game_visual_v17.1_180000_14_ob_1.mp4](images/game_visual_v17.1_180000_14_ob_1.mp4)
3. [game_visual_v17.1_180000_14_ob_2.mp4](images/game_visual_v17.1_180000_14_ob_2.mp4)
4. [game_visual_v17.1_180000_14_ob_3.mp4](images/game_visual_v17.1_180000_14_ob_3.mp4)
5. [game_visual_v17.1_180000_14_ob_4.mp4](images/game_visual_v17.1_180000_14_ob_4.mp4)


## Pre-Trained Models

You can download the pre-trained model used in this project from the following link:

[Download model_180000.pth](https://drive.google.com/file/d/1o-x5V7NHakWJkM3Y2eGZsNxsJyiDbWdI/view?usp=sharing)
