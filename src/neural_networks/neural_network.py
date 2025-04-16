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
    DenseLayerConfig,
    DynamicsNetworkConfig,
    PredictionNetworkConfig,
    RepresentationNetworkConfig,
    load_config,
)
from src.environment import Environment
from src.environments.factory import create_environment
from src.neural_networks.network_builder import (
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
        config: RepresentationNetworkConfig = load_config("config.yaml").networks.representation,
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
        self.latent_height = latent_shape[1]
        self.latent_width = latent_shape[2]

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

        # F.adaptive_avg_pool2d(x, (H, W)) transforms any (height, width) in the input to exactly (H, W)
        x = F.adaptive_avg_pool2d(x, (self.latent_height, self.latent_width))
        # x should now be of shape (B x C x H x W)
        return x


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        num_actions: int,
        config: DynamicsNetworkConfig,
        is_discrete: bool = True,
    ):
        """
        Args:
            latent_shape: (C, H, W) of the latent representation.
            num_actions: Size of the discrete action space.
            config: Contains `res_net` (list of ResBlockConfig) and `reward_net` (list of DenseLayerConfig).
        """
        super().__init__()
        self.discrete = is_discrete
        self.latent_shape = latent_shape
        self.num_actions = num_actions
        c, h, w = latent_shape

        # 1) Action embedding: we embed each action into the same dimension as a flattened latent (C*H*W)
        if self.discrete:
            # When the action is discrete, we use one-hot encoding followed by a linear layer.
            self.action_fc = nn.Linear(num_actions, c * h * w)
        else:
            # Otherwise, use an embedding layer.
            self.action_embedding = nn.Embedding(num_actions, c * h * w)

        # 2) A small linear layer to merge (latent + action) -> shape (C * H * W)
        #    (We flatten the latent, concatenate the action embedding, then re-project.)
        self.fc_merge = nn.Linear((c * h * w) + (c * h * w), c * h * w)

        # 3) Build the residual tower from config.res_net
        #    We start with in_channels = c
        self.res_blocks = nn.ModuleList()
        in_channels = c
        for res_cfg in config.res_net:
            block = ResBlock(res_cfg, in_channels=in_channels)
            self.res_blocks.append(block)
            in_channels = res_cfg.out_channels

        # If the final res block changes the channel dimension, we map it back to c
        # so the "next latent" has the same shape as the input latent.
        if in_channels != c:
            self.res_final = nn.Conv2d(in_channels, c, kernel_size=1)
        else:
            self.res_final = nn.Identity()

        # 4) Build the reward MLP from config.reward_net
        #    The input dimension for the reward MLP is c * h * w (flattened next-latent).
        self.reward_mlp, _ = build_mlp(config.reward_net, input_dim=c * h * w)

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: [B, C, H, W]
            action: [B] discrete action IDs.
        Returns:
            next_latent: [B, C, H, W]
            reward: [B, 1] (or whatever size the last layer of reward_mlp produces)
        """
        B, C, H, W = latent_state.shape
        # assert (C, H, W) == self.latent_shape, "Latent shape mismatch."

        # 1) Flatten the current latent
        latent_flat = latent_state.view(B, -1)  # [B, C*H*W]

        # 2) Obtain the action representation.
        if self.discrete:
            # One-hot encode the action and then project.
            action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()  # [B, num_actions]
            action_rep = self.action_fc(action_one_hot)  # [B, C*H*W]
        else:
            # Use an embedding layer as before.
            action_rep = self.action_embedding(action)  # [B, C*H*W]

        # 3) Merge latent + action (concatenate) -> [B, 2*C*H*W], then project down
        merged = torch.cat([latent_flat, action_rep], dim=1)  # [B, 2*C*H*W]
        merged = F.relu(self.fc_merge(merged))  # [B, C*H*W]

        # 4) Reshape back to [B, C, H, W] for the residual tower
        x = merged.view(B, C, H, W)

        # 5) Pass through each residual block
        for block in self.res_blocks:
            x = block(x)

        # 6) Possibly map back to original channel dimension c
        next_latent = self.res_final(x)  # [B, C, H, W]

        # 7) Flatten next_latent and pass to reward MLP
        flat_next = next_latent.view(B, -1)  # [B, C*H*W]
        reward = self.reward_mlp(flat_next)  # e.g. [B, 1] if last layer out_features=1

        return next_latent, reward


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        num_actions: int,
        config: PredictionNetworkConfig,
    ):
        """
        Args:
            latent_shape: (C, H, W) of the latent representation.
            num_actions: Size of the discrete action space.
            config: Contains `res_net`, `value_net`, and `policy_net`.
        """
        super().__init__()
        self.latent_shape = latent_shape
        self.num_actions = num_actions
        c, h, w = latent_shape

        # 1) Build the residual tower from config.res_net
        self.res_blocks = nn.ModuleList()
        in_channels = c
        for res_cfg in config.res_net:
            block = ResBlock(res_cfg, in_channels=in_channels)
            self.res_blocks.append(block)
            in_channels = res_cfg.out_channels

        # If the res blocks change the channel dimension, map back to `c` so we preserve shape for downstream usage
        if in_channels != c:
            self.res_final = nn.Conv2d(in_channels, c, kernel_size=1)
        else:
            self.res_final = nn.Identity()

        # 2) Build the value MLP from config.value_net
        #    The input dimension is c*h*w after flatten
        self.value_mlp, _ = build_mlp(config.value_net, input_dim=c * h * w)

        # 3) Build the policy MLP from config.policy_net
        policy_mlp_architecture = config.policy_net
        policy_mlp_architecture.append(DenseLayerConfig(out_features=num_actions, activation="softmax"))
        self.policy_mlp, _ = build_mlp(policy_mlp_architecture, input_dim=c * h * w)

    def forward(self, latent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: [B, C, H, W]
        Returns:
            policy_logits: [B, num_actions]
            value: [B, 1] (scalar value)
        """
        B, C, H, W = latent_state.shape
        assert (C, H, W) == self.latent_shape, "Latent shape mismatch."

        # 1) Pass through residual tower
        x = latent_state
        for block in self.res_blocks:
            x = block(x)

        x = self.res_final(x)  # ensure channels = c
        # Now shape = [B, c, H, W]

        # 2) Flatten
        x_flat = x.view(B, -1)  # [B, c*h*w]

        # 3) Pass to value MLP
        value = self.value_mlp(x_flat)  # e.g. [B, 1] if last layer is out_features=1

        # 4) Pass to policy MLP
        policy_logits = self.policy_mlp(x_flat)  # [B, num_actions]

        return policy_logits, value


def load_networks(
    model_folder_path: str,
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    """
    Load the neural networks from the saved files.
    """
    config = load_config("config.yaml")

    env: Environment = create_environment(config.environment)
    observation_space = env.get_observation_space()
    num_actions = len(env.get_action_space())

    repr_net = RepresentationNetwork(
        observation_space=observation_space,
        latent_shape=config.networks.latent_shape,
        config=config.networks.representation,
    )
    dyn_net = DynamicsNetwork(
        latent_shape=config.networks.latent_shape,
        num_actions=num_actions,
        config=config.networks.dynamics,
    )
    pred_net = PredictionNetwork(
        latent_shape=config.networks.latent_shape,
        num_actions=num_actions,
        config=config.networks.prediction,
    )
    repr_net.load_state_dict(torch.load(f"{model_folder_path}/repr.pth"))
    dyn_net.load_state_dict(torch.load(f"{model_folder_path}/dyn.pth"))
    pred_net.load_state_dict(torch.load(f"{model_folder_path}/pred.pth"))
    return repr_net, dyn_net, pred_net
