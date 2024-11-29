import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pickle
from collections import deque
from replay_buffer import ReplayBufferNumpy

import torch.optim as optim
import numpy as np

from collections import deque


def huber_loss(y_true, y_pred, delta=1.0):
    """
    PyTorch implementasjon av Huber-tap.
    """
    error = y_true - y_pred
    quad_error = 0.5 * error.pow(2)
    lin_error = delta * (torch.abs(error) - 0.5 * delta)
    # Kvadratisk feil hvis abs(error) < delta, ellers lineær feil
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1.0):
    """
    Beregner middelverdi av Huber-tapet.
    """
    return torch.mean(huber_loss(y_true, y_pred, delta))


class Agent:
    """
    Basis klasse for alle agenter, inkludert DeepQLearningAgent.
    """

    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """
        Initialiser agenten.
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)

        # Tilbakestill og initialiser bufferen
        self.reset_buffer()

        # Brettets grid for posisjonsrepresentasjon
        self._board_grid = np.arange(0, self._board_size**2).reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """
        Returnerer agentens gamma-verdi.
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """
        Tilbakestill bufferen.
        """
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = []  # Placeholder for replay buffer

    def get_buffer_size(self):
        """
        Returnerer gjeldende bufferstørrelse.
        """
        return len(self._buffer)

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves=None):
        """
        Legg til data i replay buffer.
        """
        self._buffer.append((board, action, reward, next_board, done))

    def save_buffer(self, file_path='', iteration=None):
        """
        Lagre bufferen til disk.
        """
        if iteration is not None:
            assert isinstance(iteration, int), "Iteration må være en integer."
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}", 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """
        Last inn buffer fra disk.
        """
        if iteration is not None:
            assert isinstance(iteration, int), "Iteration må være en integer."
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}", 'rb') as f:
            self._buffer = pickle.load(f)


class DQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNModel, self).__init__()
        # input_shape[0] = Antall kanaler (fra frames)
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

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

class DeepQLearningAgent(Agent):
    def __init__(self, board_size=10, frames=2, buffer_size=10000, gamma=0.99, n_actions=4, use_target_net=True, version=''):
        super(DeepQLearningAgent, self).__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Korrekt input_shape basert på frames og board_size
        input_shape = (frames, board_size, board_size)
        self.dqn_model = self.build_model(input_shape, n_actions).to(self.device)

        if use_target_net:
            self._target_net = self.build_model(input_shape, n_actions).to(self.device)
            self.update_target_net()
        else:
            self._target_net = None

        self.optimizer = optim.RMSprop(self.dqn_model.parameters(), lr=0.0005)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def build_model(self, input_shape, n_actions):
        return DQNModel(input_shape, n_actions)

    def _prepare_input(self, board):
        if not isinstance(board, torch.Tensor):
            board = torch.tensor(board, dtype=torch.float32, device=self.device)
        board = board / 4.0

        if board.dim() == 5:
            board = board.view(-1, *board.shape[2:])

        if board.dim() == 4:
            board = board.permute(0, 3, 1, 2)
        elif board.dim() == 3:
            board = board.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input dimension: {board.dim()}. Expected 3 or 4 dimensions.")

        return board

    def _get_model_outputs(self, board, model=None):
        if model is None:
            model = self.dqn_model
        board = self._prepare_input(board)
        with torch.no_grad():
            return model(board).cpu().numpy()

    def move(self, board, legal_moves, extra_arg=None):
        model_outputs = self._get_model_outputs(board, self.dqn_model)
        legal_outputs = np.where(legal_moves == 1, model_outputs, -np.inf)
        return np.argmax(legal_outputs, axis=1)

    def train_agent(self, batch_size=32, reward_clip=False, num_games=None):
        if len(self._buffer) < batch_size:
            return None
        sampled_data = self._buffer[:batch_size]
        s, a, r, next_s, done = zip(*sampled_data)

        if reward_clip:
            r = np.sign(r)

        s = self._prepare_input(np.stack([np.array(obs) for obs in s]))
        next_s = self._prepare_input(np.stack([np.array(obs) for obs in next_s]))

        a = torch.tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)

        q_values = self.dqn_model(s).gather(1, a)

        with torch.no_grad():
            next_q_values = self._target_net(next_s).max(1, keepdim=True)[0]
            target_q_values = r + self._gamma * next_q_values * (1 - done)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        if self._target_net is not None:
            self._target_net.load_state_dict(self.dqn_model.state_dict())

    def save_model(self, file_path, iteration=0):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.dqn_model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")
        if self._target_net is not None:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")

    def load_model(self, file_path, iteration=0):
        self.dqn_model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pth", map_location=self.device))
        if self._target_net is not None:
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pth", map_location=self.device))


class PolicyGradientModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyGradientModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 16, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_shape[0] - 6) * (input_shape[1] - 6), 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Action logits (not probabilities)



class PolicyGradientAgent(DeepQLearningAgent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False, version=''):
        super(PolicyGradientAgent, self).__init__(board_size, frames, buffer_size,
                                                  gamma, n_actions, use_target_net, version)
        self._model = PolicyGradientModel(self._input_shape, self._n_actions).to(self.device)
        self._actor_optimizer = optim.Adam(self._model.parameters(), lr=1e-6)

    def _agent_model(self):
        # Ikke nødvendig å definere en ny modell her, da modellen er initiert i __init__
        return self._model

    def train_agent(self, beta=0.1, normalize_rewards=False, num_games=1):
        """
        Tren agenten ved hjelp av policy gradient-metoden.
        
        Parameters
        ----------
        beta : float, optional
            Vekten for entropitap.
        normalize_rewards : bool, optional
            Om belønningene skal normaliseres for stabil opplæring.
        num_games : int, optional
            Antall spill i batchen.

        Returns
        -------
        loss : float
            Total tap (policy loss + entropy loss).
        """
        # Sample fra bufferen
        s, a, r, _, _, _ = self._buffer.sample(self._buffer.get_current_size())

        # Normaliser belønningene hvis spesifisert
        if normalize_rewards:
            r = (r - np.mean(r)) / (np.std(r) + 1e-8)

        # Konverter til PyTorch-tensorer
        s = self._prepare_input(s)
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)

        # Få logits fra modellen
        logits = self._model(s)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Beregn policy-tap
        action_log_probs = (a * log_probs).sum(dim=1)  # Velger logg-sannsynlighet for utførte handlinger
        policy_loss = -torch.mean(action_log_probs * r)

        # Beregn entropitap
        entropy_loss = -torch.mean(torch.sum(probs * log_probs, dim=-1))

        # Total tap
        loss = policy_loss - beta * entropy_loss

        # Tilbakestill gradienter og oppdater modell
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return loss.item()


class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2CModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_shape[0] - 4) * (input_shape[1] - 4), 64)
        self.action_logits = nn.Linear(64, n_actions)
        self.state_values = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        action_logits = self.action_logits(x)
        state_values = self.state_values(x)
        return action_logits, state_values


class AdvantageActorCriticAgent(PolicyGradientAgent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=''):
        super(AdvantageActorCriticAgent, self).__init__(board_size, frames, buffer_size,
                                                        gamma, n_actions, use_target_net, version)
        self._model = A2CModel(self._input_shape, self._n_actions).to(self.device)
        self._optimizer = optim.RMSprop(self._model.parameters(), lr=5e-4)

    def _agent_model(self):
        return self._model

    def train_agent(self, beta=0.001, normalize_rewards=False, num_games=1):
        """
        Tren agenten ved hjelp av Advantage Actor-Critic (A2C)-metoden.
        
        Parameters
        ----------
        beta : float, optional
            Vekt for entropitap.
        normalize_rewards : bool, optional
            Om belønningene skal normaliseres.
        num_games : int, optional
            Antall spill i batchen.

        Returns
        -------
        loss : float
            Total tap (kombinasjon av policy og verdi-tap).
        """
        # Hent data fra replay buffer
        s, a, r, next_s, done, _ = self._buffer.sample(self._buffer.get_current_size())
        s = self._prepare_input(s)
        next_s = self._prepare_input(next_s)

        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Normaliser belønninger
        if normalize_rewards:
            r = (r - r.mean()) / (r.std() + 1e-8)

        # Beregn policy og verdi fra modellen
        action_logits, state_values = self._model(s)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_action_probs = torch.log_softmax(action_logits, dim=-1)

        with torch.no_grad():
            _, next_state_values = self._model(next_s)
            next_state_values = next_state_values.squeeze(-1)

        # Beregn fordelen (advantage)
        future_rewards = r + self._gamma * next_state_values * (1 - done)
        advantage = future_rewards - state_values.squeeze(-1)

        # Policy-tap
        log_probs = (log_action_probs * a).sum(dim=-1)
        policy_loss = -torch.mean(log_probs * advantage.detach())

        # Entropitap for utforskning
        entropy_loss = -torch.mean(torch.sum(action_probs * log_action_probs, dim=-1))

        # Verdi-tap
        value_loss = nn.SmoothL1Loss()(state_values.squeeze(-1), future_rewards)

        # Total tap
        loss = policy_loss - beta * entropy_loss + value_loss

        # Tilbakestill gradienter, beregn og oppdater
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item()
        }

    def update_target_net(self):
        """Oppdater vektene til målnettverket."""
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def save_model(self, file_path, iteration=0):
        """Lagre modellen til disk."""
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")

    def load_model(self, file_path, iteration=0):
        """Last inn modellen fra disk."""
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pth", map_location=self.device))



class HamiltonianCycleAgent(Agent):
    """Agent som følger en Hamiltonsk syklus for å navigere på brettet."""

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=False, version=''):
        assert board_size % 2 == 0, "Brettstørrelsen må være et partall for Hamiltonsk syklus."
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                         gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                         version=version)
        self._get_cycle_square()

    def _get_neighbors(self, point):
        """Hent naboer til et gitt punkt."""
        row, col = point // self._board_size, point % self._board_size
        neighbors = []
        for delta_row, delta_col in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if 1 <= new_row <= self._board_size - 2 and 1 <= new_col <= self._board_size - 2:
                neighbors.append(new_row * self._board_size + new_col)
        return neighbors

    def _get_cycle_square(self):
        """Beregn en Hamiltonsk syklus for et kvadratisk brett."""
        self._cycle = np.zeros(((self._board_size - 2) ** 2,), dtype=np.int64)
        index = 0
        sp = 1 * self._board_size + 1
        while index < self._cycle.shape[0]:
            self._cycle[index] = sp
            if (sp % self._board_size) % 2 == 1:  # Gå ned
                sp = ((sp // self._board_size) + 1) * self._board_size + (sp % self._board_size)
                if sp // self._board_size == self._board_size - 1:  # Juster til høyre
                    sp = ((sp // self._board_size) - 1) * self._board_size + ((sp % self._board_size) + 1)
            else:  # Gå opp
                sp = ((sp // self._board_size) - 1) * self._board_size + (sp % self._board_size)
                if sp // self._board_size == 1:  # Juster til høyre
                    sp = ((sp // self._board_size) + 1) * self._board_size + ((sp % self._board_size) + 1)
            index += 1

    def move(self, board, legal_moves, values):
        """Bestem handling basert på syklus."""
        cy_len = (self._board_size - 2) ** 2
        curr_head = np.sum(self._board_grid * (board[:, :, 0] == values['head']).reshape(self._board_size, self._board_size))
        index = 0
        while self._cycle[index] != curr_head:
            index = (index + 1) % cy_len
        next_head = self._cycle[(index + 1) % cy_len]

        # Beregn neste handling
        curr_row, curr_col = self._point_to_row_col(curr_head)
        next_row, next_col = self._point_to_row_col(next_head)
        dx, dy = next_col - curr_col, next_row - curr_row
        if dx == 1 and dy == 0:
            return 0  # Høyre
        elif dx == -1 and dy == 0:
            return 2  # Venstre
        elif dx == 0 and dy == 1:
            return 1  # Ned
        elif dx == 0 and dy == -1:
            return 3  # Opp
        return -1  # Feil

    def get_action_proba(self, board, values):
        """For kompatibilitet: Returner handling som sannsynlighetsfordeling."""
        move = self.move(board, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        """For kompatibilitet."""
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        """For kompatibilitet."""
        pass

  



class SupervisedLearningAgent(DeepQLearningAgent):
    """
    Agent som bruker overvåket læring til å trene en klassifikasjonsmodell
    basert på handlingene tatt av en perfekt agent.
    """

    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=''):
        super(SupervisedLearningAgent, self).__init__(board_size=board_size, frames=frames, 
                                                      buffer_size=buffer_size, gamma=gamma, 
                                                      n_actions=n_actions, use_target_net=use_target_net, 
                                                      version=version)
        # Softmax-utdata for klassifisering
        self._classification_model = nn.Sequential(
            self._model,  # DQN-modellen (arvet fra DeepQLearningAgent)
            nn.Softmax(dim=-1)  # Softmax for klassifikasjon
        )
        self._optimizer = optim.Adam(self._classification_model.parameters(), lr=0.0005)
        self._loss_fn = nn.CrossEntropyLoss()

    def train_agent(self, batch_size=32, num_games=1, epochs=5, reward_clip=False):
        """
        Tren modellen ved hjelp av overvåket læring.

        Parameters
        ----------
        batch_size : int, optional
            Antall eksempler som hentes fra buffer.
        num_games : int, optional
            Ikke brukt her, inkludert for konsistens.
        epochs : int, optional
            Antall epoker å trene modellen for.
        reward_clip : bool, optional
            Ikke brukt her, inkludert for konsistens.

        Returns
        -------
        loss : float
            Gjennomsnittlig tap fra den siste epoken.
        """
        for epoch in range(epochs):
            s, a, _, _, _, _ = self._buffer.sample(batch_size)
            s = self._prepare_input(s)
            a = torch.tensor(a, dtype=torch.long, device=self.device)  # Handlingene som mål

            # Fremoverpass
            self._optimizer.zero_grad()
            logits = self._classification_model(s)
            loss = self._loss_fn(logits, a.argmax(dim=-1))  # Krysstap for klassifisering

            # Tilbakepropagering
            loss.backward()
            self._optimizer.step()

        return loss.item()

    def get_max_output(self):
        """
        Få maksimumsverdien av Q-verdier fra modellen. Brukes til
        å normalisere outputlagene for DQN.

        Returns
        -------
        max_value : float
            Maksimumsverdien produsert av nettverket.
        """
        s, _, _, _, _, _ = self._buffer.sample(self.get_buffer_size())
        s = self._prepare_input(s)
        with torch.no_grad():
            q_values = self._model(s)
        max_value = q_values.abs().max().item()
        return max_value

    def normalize_layers(self, max_value=None):
        """
        Normaliser vektene i outputlaget ved å dele på en maksimumsverdi.

        Parameters
        ----------
        max_value : float, optional
            Verdien å dele vektene på. Standard er 1 hvis None.
        """
        if max_value is None or np.isnan(max_value):
            max_value = 1.0

        # Hent vektene til det siste laget
        with torch.no_grad():
            last_layer = self._model.action_logits
            weights, biases = last_layer.weight, last_layer.bias
            last_layer.weight.copy_(weights / max_value)
            last_layer.bias.copy_(biases / max_value)



class BreadthFirstSearchAgent(Agent):
    """
    Agent som finner den korteste veien fra slangens hode til maten
    ved å bruke bredde-først-søk, og unngår hindringer.
    """

    def _get_neighbors(self, point, values, board):
        """
        Hent naboene til et gitt punkt, gitt brettet og verdiene.

        Parameters
        ----------
        point : int
            Punktets posisjon som en enkelt int (flate indekser).
        values : dict
            Verdier for brettobjekter (som 'head', 'food', 'board').
        board : np.array
            Gjeldende bretttilstand.

        Returns
        -------
        neighbors : list
            Liste over naboindekser som kan besøkes.
        """
        row, col = self._point_to_row_col(point)
        neighbors = []
        for delta_row, delta_col in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            new_row, new_col = row + delta_row, col + delta_col
            if board[new_row][new_col] in [values['board'], values['food'], values['head']]:
                neighbors.append(new_row * self._board_size + new_col)
        return neighbors

    def _get_shortest_path(self, board, values):
        """
        Finn den korteste veien fra slangens hode til maten ved hjelp av BFS.

        Parameters
        ----------
        board : np.array
            Gjeldende bretttilstand.
        values : dict
            Verdier for brettobjekter.

        Returns
        -------
        path : list
            Liste over punkter som representerer den korteste veien.
        """
        board = board[:, :, 0]
        head = (self._board_grid * (board == values['head'])).sum()
        points_to_search = deque([head])
        path = []
        distances = np.full((self._board_size, self._board_size), np.inf)
        visited = np.zeros((self._board_size, self._board_size))
        curr_row, curr_col = self._point_to_row_col(head)
        distances[curr_row][curr_col] = 0
        visited[curr_row][curr_col] = 1
        found = False

        while points_to_search and not found:
            curr_point = points_to_search.popleft()
            curr_row, curr_col = self._point_to_row_col(curr_point)
            neighbors = self._get_neighbors(curr_point, values, board)

            for p in neighbors:
                row, col = self._point_to_row_col(p)
                if distances[row][col] > distances[curr_row][curr_col] + 1:
                    distances[row][col] = distances[curr_row][curr_col] + 1
                if board[row][col] == values['food']:
                    found = True
                    break
                if not visited[row][col]:
                    visited[row][col] = 1
                    points_to_search.append(p)

        # Gjenopprett veien bakover fra maten til hodet
        food = (self._board_grid * (board == values['food'])).sum()
        curr_point = food
        path.append(curr_point)

        while distances[self._point_to_row_col(curr_point)] > 0:
            curr_row, curr_col = self._point_to_row_col(curr_point)
            neighbors = self._get_neighbors(curr_point, values, board)
            for p in neighbors:
                row, col = self._point_to_row_col(p)
                if distances[row][col] == distances[curr_row][curr_col] - 1:
                    path.append(p)
                    curr_point = p
                    break

        return path[::-1]

    def move(self, board, legal_moves, values):
        """
        Få handlingen for å navigere slangens hode til maten.

        Parameters
        ----------
        board : np.array
            Gjeldende bretttilstand.
        legal_moves : np.array
            Liste over lovlige handlinger.
        values : dict
            Verdier for brettobjekter.

        Returns
        -------
        a : int
            Handlingen for å flytte slangens hode.
        """
        board_main = board.copy()
        a = np.zeros((board.shape[0],), dtype=np.uint8)

        for i in range(board.shape[0]):
            board = board_main[i, :, :, :]
            path = self._get_shortest_path(board, values)
            if not path:
                a[i] = 1  # Default handling hvis ingen vei finnes
                continue

            curr_head = (self._board_grid * (board[:, :, 0] == values['head'])).sum()
            next_head = path[-2]

            curr_head_row, curr_head_col = self._point_to_row_col(curr_head)
            next_head_row, next_head_col = self._point_to_row_col(next_head)
            dx, dy = next_head_col - curr_head_col, next_head_row - curr_head_row

            if dx == 1 and dy == 0:
                a[i] = 0  # Høyre
            elif dx == -1 and dy == 0:
                a[i] = 2  # Venstre
            elif dx == 0 and dy == 1:
                a[i] = 1  # Ned
            elif dx == 0 and dy == -1:
                a[i] = 3  # Opp

        return a

    def get_action_proba(self, board, values):
        """
        For kompatibilitet: Returner handling som sannsynlighetsfordeling.
        """
        move = self.move(board, None, values)
        prob = [0] * self._n_actions
        prob[move] = 1
        return prob

    def _get_model_outputs(self, board=None, model=None):
        """For kompatibilitet."""
        return [[0] * self._n_actions]

    def load_model(self, **kwargs):
        """For kompatibilitet."""
        pass

