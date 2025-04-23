import datetime
import os
import random
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange

import wandb
from src.config.config_loader import EnvironmentConfig, TrainingConfig
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.replay_buffer import ReplayBuffer
from src.ring_buffer import Frame, FrameRingBuffer, make_history_tensor
from src.training_data_generator import Episode

FOLDER_REGEX = re.compile(r"^(\d+)_(\d{8}_\d{6})$")
BASE_PATH = "training_runs"

# Set the random seed for reproducibility
random.seed(0)


class NeuralNetworkManager:
    def __init__(
        self,
        config: TrainingConfig,
        repr_net: RepresentationNetwork,
        dyn_net: DynamicsNetwork,
        pred_net: PredictionNetwork,
        device="cpu",
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

        self.optimizer = torch.optim.SGD(
            list(self.repr_net.parameters()) + list(self.dyn_net.parameters()) + list(self.pred_net.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            # betas=config.betas,
        )

        self.device = device

    def train(self, replay_buffer: ReplayBuffer) -> float:
        """
        Train MuZero's neural networks using Backpropagation Through Time (BPTT).
        After training, automatically saves the model in /models/<counter>_<datetime>/,
        where <counter> is the next available integer after scanning existing folders.
        """
        batch_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        for mb in trange(self.mbs):
            current_batch, indices = replay_buffer.sample_batch()

            current_batch = self._filter_for_valid_episodes(current_batch)

            self.optimizer.zero_grad()
            # We'll keep track of each episode's "priority error"
            errors_for_priorities = []
            ep_indices_used = []

            batch_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

            for local_idx, episode in enumerate(current_batch):
                # local_idx is the index within the mini-batch,
                # but we also want the "global" index from 'indices' we got from the buffer.
                global_idx = indices[local_idx]

                max_k = len(episode.chunks) - (self.roll_ahead + 1)
                if max_k < 0:
                    # Not enough steps to do roll_ahead; skip
                    continue

                k = random.randrange(max_k + 1)
                start_roll_idx = max(0, k - self.lookback)

                # Gather data from the chosen segment
                Sb_k = [episode.chunks[i].state for i in range(start_roll_idx, k + 1)]
                Ab_k = [episode.chunks[i].best_action for i in range(k, k + self.roll_ahead)]
                Pb_k = [episode.chunks[i].policy for i in range(k, k + self.roll_ahead + 1)]
                Vb_k = [episode.chunks[i].value for i in range(k, k + self.roll_ahead + 1)]
                Rb_k = [episode.chunks[i + 1].reward for i in range(k, k + self.roll_ahead)]

                # Compute loss for this chunk
                step_loss = self.bptt(Sb_k, Ab_k, (Pb_k, Vb_k, Rb_k))
                batch_loss += step_loss

                # Convert step_loss to float for priority
                priority_err = step_loss.detach().abs().item()
                errors_for_priorities.append(priority_err)
                ep_indices_used.append(global_idx)

            # Average loss across the mini-batch
            effective_batch_size = len(current_batch)
            if effective_batch_size > 0:
                batch_loss = batch_loss / effective_batch_size
                # Now do a single backward pass for the entire mini-batch
                batch_loss.backward()
                self.optimizer.step()
                self.loss_history.append(batch_loss.item())
                wandb.log({"batch_loss": batch_loss.item()})

                # Update replay buffer priorities
                replay_buffer.update_priorities(ep_indices_used, errors_for_priorities)

        return batch_loss.item()

    def _filter_for_valid_episodes(self, episode_history: list[Episode]) -> list[Episode]:
        valid_episodes = []
        for episode in episode_history:
            # Check if the episode has enough chunks for training
            if len(episode.chunks) >= self.lookback + self.roll_ahead + 1:
                valid_episodes.append(episode)
            else:
                print(
                    f"Skipping episode with {len(episode.chunks)} chunks; "
                    f"requires at least {self.lookback + self.roll_ahead + 1}."
                )
        return valid_episodes

    def bptt(self, Sb_k: list[Tensor], Ab_k: list[Tensor], PVR: tuple) -> torch.Tensor:
        """
        Perform Backpropagation Through Time (BPTT) on MuZero's three networks.

        Args:
            Sb_k (list[Tensor]): States from [k - lookback ... k].
            Ab_k (list[Tensor]): Actions for roll_ahead steps k..k+w-1.
            PVR (tuple): ([π], [v], [r]) from k..k+w (policies & values),
                         and k+1..k+w (rewards).
        Returns:
            torch.Tensor: The total (summed) loss across unrolled steps.
        """
        Pb_k, Vb_k, Rb_k = PVR

        history_frames = FrameRingBuffer(size=self.repr_net.history_length)
        history_frames.fill(Frame(state=Sb_k[0], action=Ab_k[0]))
        for i in range(1, len(Sb_k)):
            history_frames.add(Frame(state=Sb_k[i], action=Ab_k[i]))

        latent_state = self.repr_net(make_history_tensor(history_frames))

        total_loss = torch.zeros(1, dtype=torch.float32).to(self.device)

        # Unroll for self.roll_ahead steps
        for i in range(self.roll_ahead):
            # Predict policy and value from current latent_state
            pred_policy, pred_value = self.pred_net(latent_state)

            # Convert the single-step targets to tensors if needed
            target_policy = Pb_k[i]
            target_value = Vb_k[i]
            target_reward = Rb_k[i]

            if not isinstance(target_policy, torch.Tensor):
                target_policy = torch.tensor(target_policy, dtype=torch.float32).to(self.device)
            if not isinstance(target_value, torch.Tensor):
                target_value = torch.tensor([target_value], dtype=torch.float32).to(self.device)
            if not isinstance(target_reward, torch.Tensor):
                target_reward = torch.tensor([target_reward], dtype=torch.float32).to(self.device)

            # Calculate the next latent state and reward
            action_i = Ab_k[i]
            if not isinstance(action_i, torch.Tensor):
                action_i = torch.tensor([action_i], dtype=torch.long).to(self.device)
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
            final_target_value = torch.tensor([final_target_value], dtype=torch.float32).to(self.device)
        _, final_pred_value = self.pred_net(latent_state)

        # No policy or reward for the final step, only a value
        final_PVR = ([], [final_target_value], [])
        final_step_loss = self.loss(final_PVR, reward=None, value=final_pred_value, policy=None)
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
        policy_loss_val = torch.zeros(1, dtype=torch.float32).to(self.device)
        value_loss_val = torch.zeros(1, dtype=torch.float32).to(self.device)
        reward_loss_val = torch.zeros(1, dtype=torch.float32).to(self.device)

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
        wandb.log(
            {
                "policy_loss": policy_loss_val.item(),
                "value_loss": value_loss_val.item(),
                "reward_loss": reward_loss_val.item(),
            }
        )
        return policy_loss_val + value_loss_val + reward_loss_val

    def reward_loss(self, target_reward: Tensor, pred_reward: Tensor) -> Tensor:
        """Compute the MSE between target reward and predicted reward."""
        return F.mse_loss(pred_reward, target_reward) * self.config.reward_coefficient

    def value_loss(self, target_value: Tensor, pred_value: Tensor) -> Tensor:
        """Compute the MSE between target value and predicted value."""
        return F.mse_loss(pred_value, target_value) * self.config.value_coefficient

    def policy_loss(self, target_policy: Tensor, pred_policy: Tensor) -> Tensor:
        """
        Compute cross-entropy or KL divergence for the policy:
        l_p(π, p) = - sum( π * log p ).
        """
        return F.cross_entropy(pred_policy, target_policy) * self.config.policy_coefficient

    def save_models(
        self,
        final_loss_val: float = 0.0,
        env_config: EnvironmentConfig = None,
        show_plot: bool = True,
    ) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
        """
        Save the neural networks to disk in /models/<counter>_<datetime>/.
        Also create a .txt file with hyperparameters and the final loss.
        """
        counter = self._get_next_model_counter(BASE_PATH)

        # Create folder name: e.g. "0_20250324_134501"
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{counter}_{now_str}"
        model_path = os.path.join(BASE_PATH, "models/" + folder_name)
        os.makedirs(model_path, exist_ok=True)

        torch.save(self.repr_net.state_dict(), os.path.join(model_path, "repr.pth"))
        torch.save(self.dyn_net.state_dict(), os.path.join(model_path, "dyn.pth"))
        torch.save(self.pred_net.state_dict(), os.path.join(model_path, "pred.pth"))

        # Write hyperparameters and final loss into a text file
        info_path = os.path.join(BASE_PATH, "info/" + folder_name)
        os.makedirs(info_path, exist_ok=True)
        txt_path = os.path.join(info_path, "training_info.txt")
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
            f.write("Network configuration:\n\n")
            f.write(f"{self.repr_net}\n\n")
            f.write(f"{self.dyn_net}\n\n")
            f.write(f"{self.pred_net}\n\n")

        print(f"Saved model to: {model_path}/")

        self.plot_loss(info_path, show_plot)
        return self.repr_net, self.dyn_net, self.pred_net

    def load_model(self, counter: int) -> None:
        """
        Load the neural networks from the *newest* subfolder matching <counter>_<datetime>.

        Args:
            counter (int): Numerical counter used in the folder name (e.g. 7).
        """
        base_path = os.path.join(BASE_PATH, "models")
        subfolders = os.listdir(base_path)
        # Filter out those matching <counter>_<datetime>
        matching = []
        for folder in subfolders:
            match = FOLDER_REGEX.match(folder)
            if match:
                c_val = int(match.group(1))
                dt_str = match.group(2)
                if c_val == counter:
                    matching.append((folder, dt_str))

        if not matching:
            raise FileNotFoundError(f"No folder found in '{base_path}' with counter = {counter}")

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
        Look for all subfolders in `base_path/models` and `base_path/info` matching the pattern <counter>_<datetime>,
        find the maximum <counter> so far, and return max+1. If none exist, return 0.
        """
        models_path = os.path.join(base_path, "models")
        info_path = os.path.join(base_path, "info")

        # Ensure both directories exist
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
        if not os.path.isdir(info_path):
            os.makedirs(info_path)

        # Collect subfolders from both paths
        subfolders = os.listdir(models_path) + os.listdir(info_path)
        max_counter = -1
        for folder in subfolders:
            match = FOLDER_REGEX.match(folder)
            if match:
                c_val = int(match.group(1))
                max_counter = max(max_counter, c_val)

        return max_counter + 1

    def plot_loss(self, path: str, shall_show: bool) -> None:
        """Plot the loss history during training."""
        img_path = os.path.join(path, "training_loss.png")
        plt.plot(self.loss_history)
        plt.xlabel("Mini-batch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(img_path)
        if shall_show:
            plt.show()
