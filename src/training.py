import random
import torch
import torch.nn.functional as F
from torch import Tensor
from src.environment import Environment
from src.neural_network import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from src.training_data_generator import Episode

class NeuralNetworkManager:
    def __init__(self, config: dict,
                 repr_net: RepresentationNetwork,
                 dyn_net: DynamicsNetwork,
                 pred_net: PredictionNetwork):
        """
        Args:
            config (dict): Contains hyperparameters like 'lookback', 'roll_ahead', 'learning_rate'
            repr_net, dyn_net, pred_net: The three MuZero networks
        """
        self.lookback = config["lookback"]
        self.roll_ahead = config["roll_ahead"]
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net

        self.optimizer = torch.optim.Adam(
            list(self.repr_net.parameters()) +
            list(self.dyn_net.parameters()) +
            list(self.pred_net.parameters()),
            lr=config["learning_rate"]
        )

    def train(self, episode_history: list[Episode], mbs: int):
        """
        Train MuZero's neural networks using Backpropagation Through Time (BPTT).

        Args:
            episode_history (list[Episode]): A list of episodes.
            mbs (int): The mini-batch size (number of BPTT updates).
        """
        for _ in range(mbs):
            # Randomly pick an episode
            b = random.randrange(len(episode_history))
            episode = episode_history[b]

            # Sample a valid starting index for unrolling
            max_k = len(episode.states) - (self.roll_ahead + 1)
            if max_k < 0:
                # Not enough steps in this episode to do roll_ahead
                continue
            k = random.randrange(max_k + 1)
            start_idx = max(0, k - self.lookback)

            # Gather data
            Sb_k = [episode.states[i].state for i in range(start_idx, k + 1)]
            Ab_k = [episode.states[i].best_action for i in range(k, k + self.roll_ahead)]
            Πb_k = [episode.states[i].policy for i in range(k, k + self.roll_ahead + 1)]
            Vb_k = [episode.states[i].value  for i in range(k, k + self.roll_ahead + 1)]
            Rb_k = [episode.states[i+1].reward for i in range(k, k + self.roll_ahead)]

            # Optimize
            self.optimizer.zero_grad()
            total_loss = self.bptt(Sb_k, Ab_k, (Πb_k, Vb_k, Rb_k))
            total_loss.backward()
            self.optimizer.step()

    def bptt(self, Sb_k: list[Environment],
             Ab_k: list[Tensor],
             PVR: tuple) -> torch.Tensor:
        """
        Perform Backpropagation Through Time (BPTT) on MuZero's three networks.

        Args:
            Sb_k (list[Environment]): States from [k - lookback ... k].
            Ab_k (list[Tensor]): Actions for roll_ahead steps k..k+w-1.
            PVR (tuple): ([π], [v], [r]) from k..k+w (policies & values),
                         and k+1..k+w (rewards).
        Returns:
            torch.Tensor: The total (summed) loss across unrolled steps.
        """
        Πb_k, Vb_k, Rb_k = PVR

        last_real_state = Sb_k[-1]
        if not isinstance(last_real_state, torch.Tensor):
            last_real_state = torch.tensor(last_real_state, dtype=torch.float32)
        latent_state = self.repr_net(last_real_state.unsqueeze(0))

        total_loss = torch.zeros(1, dtype=torch.float32)

        # Unroll for self.roll_ahead steps
        for i in range(self.roll_ahead):
            # Predict policy and value from current latent_state
            pred_policy, pred_value = self.pred_net(latent_state)

            # Convert the single-step targets to tensors if needed
            target_policy = Πb_k[i]
            target_value  = Vb_k[i]
            target_reward = Rb_k[i]

            if not isinstance(target_policy, torch.Tensor):
                target_policy = torch.tensor(target_policy, dtype=torch.float32)
            if not isinstance(target_value, torch.Tensor):
                target_value = torch.tensor(target_value, dtype=torch.float32)
            if not isinstance(target_reward, torch.Tensor):
                target_reward = torch.tensor(target_reward, dtype=torch.float32)

            # Calculate the next latent state and reward
            action_i = Ab_k[i]
            if not isinstance(action_i, torch.Tensor):
                action_i = torch.tensor(action_i, dtype=torch.long)
            next_latent_state, pred_reward = self.dyn_net(latent_state, action_i)

            # Single-step "PVR"
            single_step_PVR = ([target_policy], [target_value], [target_reward])

            # Compute single-step loss
            step_loss = self.loss(
                single_step_PVR,
                pred_reward,
                pred_value,
                pred_policy
            )
            total_loss += step_loss

            # Move to the next latent state
            latent_state = next_latent_state

        # Final Value at step k + roll_ahead
        final_target_value = Vb_k[self.roll_ahead]
        if not isinstance(final_target_value, torch.Tensor):
            final_target_value = torch.tensor(final_target_value, dtype=torch.float32)
        _, final_pred_value = self.pred_net(latent_state)

        # No policy or reward for the final step, only a value
        final_PVR = ([], [final_target_value], [])
        final_step_loss = self.loss(
            final_PVR,
            reward=None,
            value=final_pred_value,
            policy=None
        )
        total_loss += final_step_loss

        return total_loss

    def loss(self, PVR, reward, value, policy):
        """
        Calculate the loss of the neural networks.

        Args:
            PVR (tuple): A tuple of (policies, values, rewards), each a *list* even if length=1.
            reward (Tensor|None): The predicted reward from the dynamics network.
            value (Tensor|None):  The predicted value from the prediction network.
            policy (Tensor|None): The predicted policy from the prediction network.

        Returns:
            torch.Tensor: The summed loss for policy, value, and reward.
        """
        Πb_k, Vb_k, Rb_k = PVR

        # Start all sub-losses at 0
        policy_loss_val = torch.zeros(1, dtype=torch.float32)
        value_loss_val  = torch.zeros(1, dtype=torch.float32)
        reward_loss_val = torch.zeros(1, dtype=torch.float32)

        # 1) Policy loss
        if policy is not None and len(Πb_k) > 0:
            target_policy = Πb_k[0]
            # pred_policy shape: [1, num_actions]; we do policy[0] to drop the batch
            policy_loss_val = self.policy_loss(target_policy, policy[0])

        # 2) Value loss
        if value is not None and len(Vb_k) > 0:
            target_value = Vb_k[0]
            value_loss_val = self.value_loss(target_value, value[0])

        # 3) Reward loss
        if reward is not None and len(Rb_k) > 0:
            target_reward = Rb_k[0]
            reward_loss_val = self.reward_loss(target_reward, reward[0])

        # (Optional) Add L2 regularization, etc.
        return policy_loss_val + value_loss_val + reward_loss_val

    def reward_loss(self, target_reward: Tensor, pred_reward: Tensor) -> Tensor:
        """Compute the MSE between target reward and predicted reward."""
        return F.mse_loss(pred_reward, target_reward)

    def value_loss(self, target_value: Tensor, pred_value: Tensor) -> Tensor:
        """Compute the MSE between target value and predicted value."""
        return F.mse_loss(pred_value, target_value)

    def policy_loss(self, target_policy: Tensor, pred_policy: Tensor) -> Tensor:
        """Compute cross-entropy or KL divergence for policy. 
           We treat MuZero's approach: l_p(π, p) = - sum( π * log p ).
        """
        return -torch.sum(target_policy * torch.log(pred_policy))