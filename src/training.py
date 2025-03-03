import random
from src.neural_network import DynamicsNetwork, PredictionNetwork, RepresentationNetwork
from src.training_data_generator import Episode

class NeuralNetworkManager:
    def __init__(self, config: dict, repr_net: RepresentationNetwork, dyn_net: DynamicsNetwork, pred_net: PredictionNetwork):
        self.lookback = config["lookback"]
        self.roll_ahead = config["roll_ahead"]
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net

    def train(self, trinet, episode_history: list[Episode], mbs: int):
        """
        Train MuZero's neural networks using Backpropagation Through Time (BPTT).

        Args:
            trinet (MuZeroNetwork): The MuZero networks (representation, dynamic and prediction).
            episode_history (list[Episode]): A list of episodes.
            mbs (int): The mini-batch size.
        """
        for m in range(mbs):
            b = random.choice(range(len(episode_history)))
            episode = episode_history[b]
            k = random.choice(len(episode.states))

            Sb_k = [episode.states[i].state for i in range(k, k + 1 + self.lookback)]
            Ab_k = [episode.states[i].best_action for i in range(k + 1, k + 1 + self.roll_ahead)]
            Πb_k = [episode.states[i].policy for i in range(k, k + 1 + self.roll_ahead)]
            Vb_k = [episode.states[i].value for i in range(k, k + 1 + self.roll_ahead)]
            Rb_k = [episode.states[i].reward for i in range(k + 1, k + 1 + self.roll_ahead)]

            PVR = (Πb_k, Vb_k, Rb_k)
            
            # Do BPTT
            self.bptt(trinet, Sb_k, Ab_k, PVR)

    def bptt(self, trinet, Sb_k, Ab_k, PVR):
        """
        Perform Backpropagation Through Time (BPTT) on the MuZero neural networks.

        Args:
            trinet (MuZeroNetwork): The MuZero networks (representation, dynamic and prediction).
            Sb_k (list[Tensor]): A list of states.
            Ab_k (list[Tensor]): A list of actions.
            PVR (tuple): A tuple containing the policy, value and reward lists.
        """
        latent_state = self.repr_net.forward(Sb_k)
        next_latent_state, reward = self.dyn_net.forward(latent_state, Ab_k)
        policy, value = self.pred_net.forward(latent_state)
        L_k = self.loss(PVR, policy, value, reward)

    def loss(self, PVR, reward, value, policy):
        """
        Calculate the loss of the neural networks.

        Args:
            PVR (tuple): A tuple containing the policy, value and reward lists.
            policy (Tensor): The policy output of the prediction network.
            value (Tensor): The value output of the prediction network.
            reward (Tensor): The reward output of the dynamics network.
        """
        Πb_k, Vb_k, Rb_k = PVR
        reward_loss = self.reward_loss(Rb_k, reward)
        value_loss = self.value_loss(Vb_k, value)
        policy_loss = self.policy_loss(Πb_k, policy)
        L2 = 0 # TODO: Add L2 regularization
        return policy_loss + value_loss + reward_loss + L2

    def reward_loss(self, Rb_k, reward):
        """
        Calculate the reward loss.

        Args:
            Rb_k (list[float]): A list of rewards.
            reward (Tensor): The reward output of the dynamics network.
        """
        pass

    def value_loss(self, Vb_k, value):
        """
        Calculate the value loss.

        Args:
            Vb_k (list[float]): A list of values.
            value (Tensor): The value output of the prediction network.
        """
        pass

    def policy_loss(self, Πb_k, policy):
        """
        Calculate the policy loss.

        Args:
            Πb_k (list[Tensor]): A list of policies.
            policy (Tensor): The policy output of the prediction network.
        """
        pass