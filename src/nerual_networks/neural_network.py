"""
This module contains the neural network components of the model. The model consists of three components:
1. Representation Network: Maps the raw observation to an abstract latent state.
2. Dynamics Network: Predicts the next latent state and immediate reward given the current latent state and action.
3. Prediction Network: Produces a policy (actor) and a value estimate (critic) from the latent state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config_loader import (
    RepresentationNetworkConfig,
    DynamicsNetworkConfig,
    PredictionNetworkConfig,
    load_config,
)
from src.nerual_networks.network_builder import ResBlock, build_downsample_layer

latent_shape: tuple[int, int, int] = (load_config("config.yaml").networks.latent_shape,)


class RepresentationNetwork(nn.Module):
    """
    Maps the raw observation (e.g., an image) to an abstract latent state,
    using the `RepresentationNetworkConfig` which has a `downsample` list and
    a `res_net` list of Residual Blocks.
    """

    def __init__(
        self,
        observation_space: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
        config: RepresentationNetworkConfig = load_config(
            "config.yaml"
        ).networks.representation,
    ):
        """
        Args:
            observation_space (tuple): e.g. (C, H, W)
            latent_shape (tuple): e.g. (latent_channels, latent_height, latent_width)
            config (RepresentationNetworkConfig): The Pydantic config containing layer definitions.
        """
        super().__init__()

        # 1) Build the 'downsample' layers
        self.downsample_layers = nn.ModuleList()
        in_channels = observation_space[0]
        for layer_cfg in config.downsample:
            layer, out_channels = build_downsample_layer(layer_cfg, in_channels)
            self.downsample_layers.append(layer)
            in_channels = out_channels

        # 2) Build the residual blocks
        self.res_blocks = nn.ModuleList()
        for res_cfg in config.res_net:
            res_block = ResBlock(res_cfg, in_channels)
            self.res_blocks.append(res_block)
            in_channels = res_cfg.out_channels

        # 3) We often need a final transformation to get the exact latent shape desired.
        #    We can either:
        #     a) Use a conv that ensures we have latent_shape[0] out_channels
        #     b) Flatten and use a Linear to produce the exact dimension
        #    In many MuZero architectures, we keep the shape as (C,H,W).
        #    So let's do a final conv if the out_channels != latent_shape[0].
        final_out_channels = latent_shape[0]
        if in_channels != final_out_channels:
            self.final_conv = nn.Conv2d(in_channels, final_out_channels, 1)
        else:
            self.final_conv = nn.Identity()

        # We also store the expected spatial dimension (latent_shape[1], latent_shape[2]).
        # We do not strictly need to enforce it here, but you could adapt your network to do so.
        # self.latent_height = latent_shape[1]
        # self.latent_width = latent_shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all downsample layers, then all residual blocks,
        then a final conv, returning a [B, C, H, W] latent.
        """
        # 1) Downsampling path
        for layer in self.downsample_layers:
            x = layer(x)

        # 2) Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # 3) Final conv if needed
        x = self.final_conv(x)

        # x should now be of shape (B, latent_shape[0], ?, ?)
        # If you want to enforce the final H,W, you could add code to adapt or check here.
        return x


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
