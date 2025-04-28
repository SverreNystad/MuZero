import datetime
import os
import random
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import lr_scheduler
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

        # Set up the optimizer based on the chosen type
        match config.optimizer.lower():
            case "sgd":
                self.optimizer = torch.optim.SGD(
                    list(self.repr_net.parameters()) + list(self.dyn_net.parameters()) + list(self.pred_net.parameters()),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    momentum=config.momentum,
                )
            case "adam":
                self.optimizer = torch.optim.Adam(
                    list(self.repr_net.parameters()) + list(self.dyn_net.parameters()) + list(self.pred_net.parameters()),
                    lr=config.learning_rate,
                    betas=config.betas,
                    weight_decay=config.weight_decay,
                )
            case "adamw":
                self.optimizer = torch.optim.AdamW(
                    list(self.repr_net.parameters()) + list(self.dyn_net.parameters()) + list(self.pred_net.parameters()),
                    lr=config.learning_rate,
                    betas=config.betas,
                    weight_decay=config.weight_decay,
                )
            case "rmsprop":
                self.optimizer = torch.optim.RMSprop(
                    list(self.repr_net.parameters()) + list(self.dyn_net.parameters()) + list(self.pred_net.parameters()),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
            case _:
                raise ValueError(f"Unsupported optimizer type: {config.optimizer}")

        # Set up the Learning rate decay on the chosen type
        self.scheduler = None
        match config.lr_schedule.lower():
            case "step":
                self.scheduler = lr_scheduler.StepLR(
                    self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
                )
            case "multi_step":
                self.scheduler = lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=config.scheduler_milestones, gamma=config.scheduler_gamma
                )
            case "exponential":
                self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
            case "cosine_annealing":
                self.scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.scheduler_T_max, eta_min=getattr(config, "scheduler_eta_min", 0)
                )
            case "reduce_lr_on_plateau":
                self.scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience
                )

        self.device = device

    def train(self, replay_buffer: ReplayBuffer) -> float:
        """
        Train MuZero's neural networks using Backpropagation Through Time (BPTT).
        Returns average batch loss.
        """
        for mb in trange(self.mbs):
            current_batch, indices = replay_buffer.sample_batch()
            current_batch = self._filter_for_valid_episodes(current_batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Batch-level accumulators
            batch_total_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            batch_policy_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            batch_value_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            batch_reward_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            errors_for_priorities = []
            ep_indices_used = []

            for local_idx, episode in enumerate(current_batch):
                # local_idx is the index within the mini-batch,
                # but we also want the "global" index from 'indices' we got from the buffer.
                global_idx = indices[local_idx]
                max_k = len(episode.chunks) - (self.roll_ahead + 1)
                if max_k < 0:
                    # Not enough steps to do roll_ahead; skip
                    continue

                k = random.randrange(max_k + 1)
                start_idx = max(0, k - self.lookback)

                # States:   Sb,k = {s_b,k−q, …, s_b,k}
                # Actions:  Ab,k = {a_b,k+1, …, a_b,k+w}
                # Policies: Π* b,k = {π_b,k, …, π_b,k+w}
                # Values:   Z* b,k = {v* b,k, …, v* b,k+w}
                # Rewards:  R* b,k = {r* b,k+1, …, r* b,k+w}
                past_states = [episode.chunks[i].state for i in range(start_idx, k)]
                past_actions = [episode.chunks[i].best_action for i in range(start_idx, k)]
                rollout_actions = [episode.chunks[i].best_action for i in range(k, k + self.roll_ahead + 1)]
                Pb_k = [episode.chunks[i].policy for i in range(k, k + self.roll_ahead + 1)]
                Vb_k = [episode.chunks[i].value for i in range(k, k + self.roll_ahead + 1)]
                Rb_k = [episode.chunks[i].reward for i in range(k, k + self.roll_ahead + 1)]

                # Compute z targets
                Zb_k = self._compute_z_targets(Rb_k, Vb_k, self.config.discount_factor)

                # Get component losses
                p_loss, v_loss, r_loss = self.bptt(past_states, past_actions, rollout_actions, (Pb_k, Zb_k, Rb_k))
                step_loss = p_loss + v_loss + r_loss

                batch_policy_loss += p_loss
                batch_value_loss += v_loss
                batch_reward_loss += r_loss
                batch_total_loss += step_loss

                errors_for_priorities.append(step_loss.detach().abs().item())
                ep_indices_used.append(global_idx)

            if len(current_batch) > 0:
                # Average over episodes
                batch_policy_loss /= len(current_batch)
                batch_value_loss /= len(current_batch)
                batch_reward_loss /= len(current_batch)
                batch_total_loss /= len(current_batch)

                # Backprop and step
                batch_total_loss.backward()
                self.optimizer.step()

                # Scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(batch_total_loss)
                    else:
                        self.scheduler.step()

                # Log to W&B
                wandb.log(
                    {
                        "loss/batch": batch_total_loss.item(),
                        "loss/policy": batch_policy_loss.item(),
                        "loss/value": batch_value_loss.item(),
                        "loss/reward": batch_reward_loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                # Record history and priorities
                self.loss_history.append(batch_total_loss.item())
                replay_buffer.update_priorities(ep_indices_used, errors_for_priorities)

        return batch_total_loss.item()

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

    def _compute_z_targets(self, Rb_k: list[Tensor], Vb_k: list[Tensor], gamma):
        """
        Rb_k: list of length w of rewards [r*_{k+1}, ..., r*_{k+w}]
        Vb_k: list of length w+1 of values [v*_{k}, ..., v*_{k+w}]
        returns: list of length w+1 of z*_k targets
        """
        w = self.roll_ahead
        z_targets = []
        # for each unroll step k = 0..w:
        #   z*_k = sum_{i=1..w-k} gamma^{i-1} * Rb_k[k+i-1] + gamma^{w-k} * Vb_k[w]
        # but easier to always bootstrap from Vb_k[k + remaining horizon]
        for k in range(w + 1):
            # accumulate discounted rewards from r_{k+1} ... r_w
            accumulated_rewards = 0.0
            for i, r in enumerate(Rb_k[k:]):
                accumulated_rewards += (gamma ** (i - 1)) * r
            # bootstrap from Vb_k[k + (w - k)] == Vb_k[w]
            accumulated_rewards += (gamma ** (w - k)) * Vb_k[w]
            z_targets.append(accumulated_rewards)
        return z_targets

    def bptt(
        self,
        past_observations: list[Tensor],
        past_actions: list[Tensor],
        Ab_k: list[Tensor],
        PVR: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Perform Backpropagation Through Time (BPTT) on MuZero's three networks.

        Args:
            Sb_k (list[Tensor]): States from [k - lookback ... k].
            Ab_k (list[Tensor]): Actions for roll_ahead steps k..k+w-1.
            PVR (tuple): ([π], [v], [r]) from k..k+w (policies & values),
                         and k+1..k+w (rewards).
        Returns:
            Summed (policy_loss, value_loss, reward_loss).
        """
        Pb_k, Zb_k, Rb_k = PVR
        history_frames = FrameRingBuffer(size=self.repr_net.history_length)
        history_frames.fill(Frame(state=past_observations[0], action=past_actions[0]))
        for i in range(1, len(past_observations)):
            history_frames.add(Frame(state=past_observations[i], action=past_actions[i]))

        latent_state = self.repr_net(make_history_tensor(history_frames))

        total_p = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_v = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_r = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Unroll for self.roll_ahead steps
        for i in range(self.roll_ahead):
            pred_p, pred_v = self.pred_net(latent_state)
            t_p = torch.tensor(Pb_k[i], dtype=torch.float32, device=self.device)
            t_v = torch.tensor([Zb_k[i]], dtype=torch.float32, device=self.device)
            t_r = torch.tensor([Rb_k[i]], dtype=torch.float32, device=self.device)

            action = Ab_k[i]
            action = (
                action if isinstance(action, torch.Tensor) else torch.tensor([action], dtype=torch.long, device=self.device)
            )
            latent_state, pred_r = self.dyn_net(latent_state, action)

            # component losses
            p_loss = self.policy_loss(t_p, pred_p[0])
            v_loss = self.value_loss(t_v, pred_v[0])
            r_loss = self.reward_loss(t_r, pred_r[0])

            total_p += p_loss
            total_v += v_loss
            total_r += r_loss

        # final value only
        final_p, final_v = self.pred_net(latent_state)
        t_v_final = torch.tensor([Zb_k[self.roll_ahead]], dtype=torch.float32, device=self.device)
        v_final_loss = self.value_loss(t_v_final, final_v[0])
        t_p_final = torch.tensor(Pb_k[self.roll_ahead], dtype=torch.float32, device=self.device)
        p_final_loss = self.policy_loss(t_p_final, final_p[0])
        total_v += v_final_loss
        total_p += p_final_loss

        return total_p, total_v, total_r

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
