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
from src.nerual_networks.network_builder import (
    ResBlock,
    build_downsample_layer,
    build_mlp,
)

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
    """
    Predict the next latent state and immediate reward,
    given the current latent and an action.
    """

    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        action_space_size: int,
        config: DynamicsNetworkConfig = load_config("config.yaml").networks.dynamics,
    ):
        """
        Args:
            latent_shape: e.g. (C, H, W) for the latent representation
            action_space_size: number of discrete actions
            config: includes res_net + reward_net definitions (list[DenseLayerConfig])
        """
        super().__init__()

        # Flatten latent dims: C*H*W
        c, h, w = latent_shape
        self.latent_dim = c * h * w
        self.action_space_size = action_space_size

        # Input dimension for both MLPs = latent_dim + action_space_size
        input_dim = self.latent_dim + self.action_space_size

        # Build MLP for next-latent (res_net)
        self.next_latent_mlp, self.next_latent_dim = build_mlp(
            config.res_net, input_dim
        )

        # Build MLP for reward
        self.reward_mlp, self.reward_dim = build_mlp(config.reward_net, input_dim)

        # We assume the final output of next_latent_mlp is self.latent_dim,
        # so we can reshape back to (C,H,W).
        # If the config doesn’t guarantee that, you must handle or enforce it.
        if self.next_latent_dim != self.latent_dim:
            raise ValueError(
                "The final out_features of res_net MLP must match the flattened latent_dim, "
                f"but got {self.next_latent_dim} vs {self.latent_dim}."
            )

        # The reward MLP final output could be 1 or any dimension you desire.
        # Typically it’s 1 for a scalar reward.
        if self.reward_dim != 1:
            raise ValueError(
                "The final out_features of reward_net MLP should be 1 for a scalar reward. "
                f"Got {self.reward_dim}."
            )

    def forward(
        self,
        latent: torch.Tensor,  # [B, C, H, W]
        action: torch.Tensor,  # [B] integer action indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (next_latent [B, C, H, W], reward [B, 1])
        """
        B, C, H, W = latent.shape
        # 1) Flatten the latent
        latent_flat = latent.view(B, -1)

        # 2) One-hot encode action
        action_onehot = F.one_hot(action, num_classes=self.action_space_size).float()

        # 3) Combine latent + action
        combined = torch.cat(
            [latent_flat, action_onehot], dim=1
        )  # [B, latent_dim + action_space_size]

        # 4) Predict next latent (flattened)
        next_latent_flat = self.next_latent_mlp(combined)
        # Reshape to (B, C, H, W)
        next_latent = next_latent_flat.view(B, C, H, W)

        # 5) Predict reward
        reward = self.reward_mlp(combined)  # [B, 1]

        return next_latent, reward


class PredictionNetwork(nn.Module):
    """
    Produces a policy (actor) and a value estimate (critic) from the latent state.
    """

    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        action_space_size: int,
        config: PredictionNetworkConfig = load_config(
            "config.yaml"
        ).networks.prediction,
    ):
        super().__init__()
        c, h, w = latent_shape
        self.latent_dim = c * h * w
        self.action_space_size = action_space_size

        # 1) Trunk (res_net) MLP
        self.trunk, trunk_output_dim = build_mlp(config.res_net, self.latent_dim)

        # 2) Value head
        self.value_head, value_out_dim = build_mlp(config.value_net, trunk_output_dim)
        if value_out_dim != 1:
            raise ValueError(
                f"Value head should produce a scalar (1). Got out_dim={value_out_dim}."
            )

        # 3) Policy head
        self.policy_head, policy_out_dim = build_mlp(
            config.policy_net, trunk_output_dim
        )
        if policy_out_dim != action_space_size:
            raise ValueError(
                f"Policy head should produce {action_space_size} outputs (logits). "
                f"Got out_dim={policy_out_dim}."
            )

    def forward(
        self, latent: torch.Tensor  # [B, C, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (policy_logits [B, action_space_size], value [B, 1])
        """
        B, C, H, W = latent.shape

        # 1) Flatten
        x = latent.view(B, -1)

        # 2) Shared trunk
        trunk_out = self.trunk(x)

        # 3) Heads
        value = self.value_head(trunk_out)  # [B, 1]
        policy_logits = self.policy_head(trunk_out)  # [B, action_space_size]

        # You can apply softmax here or return raw logits:
        # policy_probs = F.softmax(policy_logits, dim=1)
        # Return raw logits if you prefer
        return policy_logits, value
