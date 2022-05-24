"""
Created by: louisenaud on 5/24/22 at 3:34 PM for banana_pickup.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import PrioritizedMemory, ReplayMemory
from network import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            state_size,
            action_size,
            seed=1412,
            nb_hidden=128,
            learning_rate=5e-4,
            memory_size=int(1e5),
            prioritized_memory=False,
            batch_size=64,
            gamma=0.99,
            tau=1e-3,
            small_eps=1e-5,
            update_every=4,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            nb_hidden (int): Pending
            learning_rate (float): Pending
            memory_size (int): Pending
            prioritized_memory (bool): Pending
            batch_size (int): Pending
            gamma (float): Pending
            tau (float): Pending
            update_every (int): Pending
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.prioritized_memory = prioritized_memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.small_eps = small_eps
        self.update_every = update_every
        self.losses = []

        # Q-Network
        self.qnetwork_local = QNetwork(
            self.state_size, self.action_size, layers=nb_hidden, seed=seed
        ).to(device)
        self.qnetwork_target = QNetwork(
            self.state_size, self.action_size, layers=nb_hidden, seed=seed
        ).to(device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.learning_rate
        )

        # Define memory
        if self.prioritized_memory:
            self.memory = PrioritizedMemory(self.memory_size, self.batch_size)
        else:
            self.memory = ReplayMemory(self.memory_size, self.batch_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, i):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                if self.prioritized_memory:
                    experiences = self.memory.sample(self.get_beta(i))
                else:
                    experiences = self.memory.sample()

                self.learn(experiences)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            small_e (float):
        """
        if self.prioritized_memory:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                index,
                sampling_weights,
            ) = experiences

        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.prioritized_memory:
            loss = self.mse_loss_prioritized(
                Q_expected, Q_targets, index, sampling_weights
            )
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        self.losses.append(loss)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save_model(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load_model(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))

    def get_beta(self, i, beta_start=0.4, beta_end=1, beta_growth=1.05):
        if not self.prioritized_memory:
            raise TypeError("This agent is not use prioritized memory")

        beta = min(beta_start * (beta_growth**i), beta_end)
        return beta

    def mse_loss_prioritized(self, Q_expected, Q_targets, index, sampling_weights):
        losses = (
                F.mse_loss(Q_expected, Q_targets, reduce=False).squeeze(1)
                * sampling_weights
        )
        self.memory.update_priority(index, losses + self.small_eps)
        return losses.mean()
