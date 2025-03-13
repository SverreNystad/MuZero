"""
This module contains the neural network components of the model. The model consists of three components:
1. Representation Network: Maps the raw observation to an abstract latent state.
2. Dynamics Network: Predicts the next latent state and immediate reward given the current latent state and action.
3. Prediction Network: Produces a policy (actor) and a value estimate (critic) from the latent state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self, input_channels: int, observation_space: tuple, latent_dim: int):
        """
        Maps the raw observation (e.g. an image) to an abstract latent state.
        Args:
            input_channels (int): Number of channels in the input observation.
            observation_space (tuple): Spatial dimensions like (height, width) of the observation or (x,y,z).
            latent_dim (int): Dimension of the latent state.
        """
        super(RepresentationNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the output of the convolutional layers
        conv_output_size: int = self.conv2.out_channels
        for dim in range(len(observation_space)):
            conv_output_size *= observation_space[dim]

        self.fc = nn.Linear(conv_output_size, latent_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation (Tensor): Input observation of shape (batch, channels, height, width).
        Returns:
            latent (Tensor): Abstract state representation of shape (batch, latent_dim).
        """
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        latent = torch.tanh(self.fc(x))  # tanh to bound the latent values
        return latent


class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int):
        """
        Given a latent state and an action, predicts the next latent state and an immediate reward.
        Args:
            latent_dim (int): Dimension of the latent state.
            num_actions (int): Number of discrete actions.
        """
        super(DynamicsNetwork, self).__init__()

        # Combine latent state and one-hot encoded action
        self.fc1 = nn.Linear(latent_dim + num_actions, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        # Reward head (can be a scalar for each sample)
        self.reward_head = nn.Linear(latent_dim, 1)

    def forward(
        self, latent_state: torch.Tensor, action_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state (Tensor): Current latent state of shape (batch, latent_dim).
           - action_logits (Tensor): Unnormalized log-probabilities over actions (batch, num_actions).
        Returns:
            next_latent (Tensor): Predicted next latent state (batch, latent_dim).
            reward (Tensor): Predicted immediate reward (batch, 1).
        """
        assert latent_state.size(0) == action_logits.size(0), "Batch size mismatch"
        # One-hot encode the action
        action_onehot = F.one_hot(
            action_logits, num_classes=self.fc1.in_features - latent_state.size(1)
        ).float()
        x = torch.cat([latent_state, action_onehot], dim=1)
        x = F.relu(self.fc1(x))
        next_latent = torch.tanh(self.fc2(x))
        reward = self.reward_head(x)
        return next_latent, reward


class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int):
        """
        Produces a policy (actor) and a value estimate (critic) from the latent state.
        Args:
            latent_dim (int): Dimension of the latent state.
            num_actions (int): Number of discrete actions.
        """
        super(PredictionNetwork, self).__init__()

        # Policy head: outputs logits over actions
        self.policy_head = nn.Linear(latent_dim, num_actions)

        # Value head: outputs a scalar value
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(self, latent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state (Tensor): Latent state of shape (batch, latent_dim).
        Returns:
            policy_logits (Tensor): Unnormalized log-probabilities over actions (batch, num_actions).
            value (Tensor): State value estimate (batch, 1).
        """
        policy_logits = self.policy_head(latent_state)
        policy_logits = F.softmax(policy_logits, dim=1)

        value = self.value_head(latent_state)

        return policy_logits, value
