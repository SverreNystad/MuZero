import os
import re
import datetime
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange
from src.config.config_loader import EnvironmentConfig, TrainingConfig
from src.environment import Environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.training_data_generator import Episode

FOLDER_REGEX = re.compile(r"^(\d+)_(\d{8}_\d{6})$")
BASE_PATH = "models"

class NeuralNetworkManager:
    def __init__(
        self,
        config: TrainingConfig,
        repr_net: RepresentationNetwork,
        dyn_net: DynamicsNetwork,
        pred_net: PredictionNetwork,
    ):
        """
        Args:
            config: Contains hyperparameters like 'lookback', 'roll_ahead', 'learning_rate', etc.
            repr_net, dyn_net, pred_net: The three MuZero networks
        """
        self.config = config
        self.lookback = config.look_back
        self.roll_ahead = config.roll_ahead
        self.mbs = config.mini_batch_size
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net
        self.loss_history = []


        self.optimizer = torch.optim.Adam(
            list(self.repr_net.parameters())
            + list(self.dyn_net.parameters())
            + list(self.pred_net.parameters()),
            lr=config.learning_rate,
            betas=config.betas,
        )

    def train(self, episode_history: list[Episode]):
        """
        Train MuZero's neural networks using Backpropagation Through Time (BPTT).
        After training, automatically saves the model in /models/<counter>_<datetime>/,
        where <counter> is the next available integer after scanning existing folders.
        """
        final_loss_val = 0.0

        for _ in trange(self.mbs):
            # Randomly pick an episode
            b = random.randrange(len(episode_history))
            episode = episode_history[b]

            # Sample a valid starting index for unrolling
            max_k = len(episode.chunks) - (self.roll_ahead + 1)
            if max_k < 0:
                # Not enough steps in this episode to do roll_ahead
                continue
            k = random.randrange(max_k + 1)
            start_idx = max(0, k - self.lookback)

            # Gather data
            Sb_k = [episode.chunks[i].state for i in range(start_idx, k + 1)]
            Ab_k = [episode.chunks[i].best_action for i in range(k, k + self.roll_ahead)]
            Pb_k = [episode.chunks[i].policy for i in range(k, k + self.roll_ahead + 1)]
            Vb_k = [episode.chunks[i].value for i in range(k, k + self.roll_ahead + 1)]
            Rb_k = [episode.chunks[i + 1].reward for i in range(k, k + self.roll_ahead)]

            # Optimize
            self.optimizer.zero_grad()
            total_loss = self.bptt(Sb_k, Ab_k, (Pb_k, Vb_k, Rb_k))
            total_loss.backward()
            self.optimizer.step()

            self.loss_history.append(total_loss.item())

        final_loss_val = total_loss.item()

        return final_loss_val

    def bptt(
        self, Sb_k: list[Environment], Ab_k: list[Tensor], PVR: tuple
    ) -> torch.Tensor:
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
        Pb_k, Vb_k, Rb_k = PVR

        last_real_state = Sb_k[-1]
        if not isinstance(last_real_state, torch.Tensor):
            last_real_state = torch.tensor(last_real_state, dtype=torch.float32)
        latent_state = self.repr_net(last_real_state)

        total_loss = torch.zeros(1, dtype=torch.float32)

        # Unroll for self.roll_ahead steps
        for i in range(self.roll_ahead):
            # Predict policy and value from current latent_state
            pred_policy, pred_value = self.pred_net(latent_state)

            # Convert the single-step targets to tensors if needed
            target_policy = Pb_k[i]
            target_value = Vb_k[i]
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
                action_i = torch.tensor([action_i], dtype=torch.long)
            next_latent_state, pred_reward = self.dyn_net(latent_state, action_i)

            # Single-step "PVR"
            single_step_PVR = ([target_policy], [target_value], [target_reward])

            # Compute single-step loss
            step_loss = self.loss(single_step_PVR, pred_reward, pred_value, pred_policy)
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
            final_PVR, reward=None, value=final_pred_value, policy=None
        )
        total_loss += final_step_loss

        return total_loss

    def loss(self, PVR, reward, value, policy):
        """
        Calculate the loss of the neural networks.

        Args:
            PVR (tuple): A tuple of (policies, values, rewards), each a list (even if length=1).
            reward (Tensor|None): The predicted reward from the dynamics network.
            value (Tensor|None):  The predicted value from the prediction network.
            policy (Tensor|None): The predicted policy from the prediction network.

        Returns:
            torch.Tensor: The summed loss for policy, value, and reward.
        """
        Πb_k, Vb_k, Rb_k = PVR

        policy_loss_val = torch.zeros(1, dtype=torch.float32)
        value_loss_val = torch.zeros(1, dtype=torch.float32)
        reward_loss_val = torch.zeros(1, dtype=torch.float32)

        # 1) Policy loss
        if policy is not None and len(Πb_k) > 0:
            target_policy = Πb_k[0]
            # pred_policy shape: [1, num_actions]; use policy[0] to drop the batch dimension
            policy_loss_val = self.policy_loss(target_policy, policy[0])

        # 2) Value loss
        if value is not None and len(Vb_k) > 0:
            target_value = Vb_k[0]
            value_loss_val = self.value_loss(target_value, value[0])

        # 3) Reward loss
        if reward is not None and len(Rb_k) > 0:
            target_reward = Rb_k[0]
            reward_loss_val = self.reward_loss(target_reward, reward[0])

        return policy_loss_val + value_loss_val + reward_loss_val

    def reward_loss(self, target_reward: Tensor, pred_reward: Tensor) -> Tensor:
        """Compute the MSE between target reward and predicted reward."""
        return F.mse_loss(pred_reward, target_reward)

    def value_loss(self, target_value: Tensor, pred_value: Tensor) -> Tensor:
        """Compute the MSE between target value and predicted value."""
        return F.mse_loss(pred_value, target_value)

    def policy_loss(self, target_policy: Tensor, pred_policy: Tensor) -> Tensor:
        """
        Compute cross-entropy or KL divergence for the policy:
        l_p(π, p) = - sum( π * log p ).
        """
        return -torch.sum(target_policy * torch.log(pred_policy))

    def save_models(self, final_loss_val: float = 0.0, env_config: EnvironmentConfig = None) -> None:
        """
        Save the neural networks to disk in /models/<counter>_<datetime>/.
        Also create a .txt file with hyperparameters and the final loss.
        """
        counter = self._get_next_model_counter(BASE_PATH)

        # Create folder name: e.g. "0_20250324_134501"
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{counter}_{now_str}"
        full_path = os.path.join(BASE_PATH, folder_name)
        os.makedirs(full_path, exist_ok=True)

        torch.save(self.repr_net.state_dict(), os.path.join(full_path, "repr.pth"))
        torch.save(self.dyn_net.state_dict(), os.path.join(full_path, "dyn.pth"))
        torch.save(self.pred_net.state_dict(), os.path.join(full_path, "pred.pth"))

        # Write hyperparameters and final loss into a text file
        txt_path = os.path.join(full_path, "training_info.txt")
        with open(txt_path, "w") as f:
            f.write("MuZero Training Info\n")
            f.write(f"Environment: {env_config.type}\n")
            f.write(f"Model saved at: {now_str}\n\n")
            f.write("Hyperparameters:\n")
            f.write(f"lookback: {self.config.look_back}\n")
            f.write(f"roll_ahead: {self.config.roll_ahead}\n")
            f.write(f"mini_batch_size: {self.config.mini_batch_size}\n")
            f.write(f"learning_rate: {self.config.learning_rate}\n")
            f.write(f"betas: {self.config.betas}\n\n")
            f.write(f"Final Loss: {final_loss_val}\n\n")
            f.write(f"Network configuration:\n\n")
            f.write(f"{self.repr_net}\n\n")
            f.write(f"{self.dyn_net}\n\n")
            f.write(f"{self.pred_net}\n\n")

        print(f"Saved model to: {full_path}/")

        self.plot_loss(full_path)

    def load_model(self, base_path: str, counter: int) -> None:
        """
        Load the neural networks from the *newest* subfolder matching <counter>_<datetime>.

        Args:
            base_path (str): The base path where models are saved (e.g. "models").
            counter (int): Numerical counter used in the folder name (e.g. 7).
        """
        subfolders = os.listdir(base_path)
        # Filter out those matching <counter>_<datetime>
        matching = []
        for folder in subfolders:
            match = FOLDER_REGEX.match(folder)
            if match:
                c_val = int(match.group(1))
                dt_str = match.group(2)  # e.g. '20250324_134501'
                if c_val == counter:
                    matching.append((folder, dt_str))

        if not matching:
            raise FileNotFoundError(
                f"No folder found in '{base_path}' with counter = {counter}"
            )

        # Sort by dt_str descending so the newest is first
        matching.sort(key=lambda x: x[1], reverse=True)
        latest_folder = matching[0][0]  # e.g. "7_20250324_134501"

        full_path = os.path.join(base_path, latest_folder)
        repr_path = os.path.join(full_path, "repr.pth")
        dyn_path = os.path.join(full_path, "dyn.pth")
        pred_path = os.path.join(full_path, "pred.pth")

        self.repr_net.load_state_dict(torch.load(repr_path))
        self.dyn_net.load_state_dict(torch.load(dyn_path))
        self.pred_net.load_state_dict(torch.load(pred_path))

        print(f"Loaded model from: {full_path}/")

    def _get_next_model_counter(self, base_path: str) -> int:
        """
        Look for all subfolders in `base_path` matching the pattern <counter>_<datetime>,
        find the maximum <counter> so far, and return max+1. If none exist, return 0.
        """
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        subfolders = os.listdir(base_path)
        max_counter = -1
        for folder in subfolders:
            match = FOLDER_REGEX.match(folder)
            if match:
                c_val = int(match.group(1))
                max_counter = max(max_counter, c_val)

        return max_counter + 1

    def plot_loss(self, path: str) -> None:
        """Plot the loss history during training."""
        img_path = os.path.join(path, "training_loss.png")
        plt.plot(self.loss_history)
        plt.xlabel("Mini-batch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(img_path)
        plt.show()
