from typing import Any, Protocol

from torch import Tensor


class Environment(Protocol):
    """
    The Environment protocol defines an interface similar to Gymnasium’s
    environment API for reinforcement learning tasks. It encapsulates
    an environment with potentially arbitrary underlying dynamics.

    The main methods are:
    - step(action): Updates the environment by one timestep given an action.
        Returns observation, reward, and whether the environment is done.
    - reset(): Resets the environment to an initial (start) state.
        Returns the initial observation.
    - render(mode): Renders the environment for visualization or debugging.
    - close(): Cleans up resources used by the environment.
    """

    def get_action_space(self) -> tuple[int, ...]:
        """
        The action space of the environment.
        """
        ...

    def get_observation_space(self) -> tuple[int, ...]:
        """
        The observation space of the environment.
        """
        ...

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        """
        Run one timestep of the environment’s dynamics.

        Args:
            action (int): The action taken by the agent.

        Returns:
            state (Any): The next observation from the environment.
            reward (float): The reward for the taken action.
            done (bool): Whether the episode has ended.gym connect four

        Note:
            In a Gymnasium environment, this function often returns an additional
            'info' dictionary. You can include that if desired.
        """
        ...

    def get_state(self) -> Tensor:
        """
        Get the current state of the environment as a PyTorch tensor.

        Returns:
            state (Tensor): The current state of the environment.
        """
        ...

    def reset(self) -> Tensor:
        """
        Reset the environment to its initial state.

        Returns:
            observation (Tensor): The initial observation for the agent.
        """
        ...

    def render(self) -> Any:
        """
        Render the environment in a human or machine-readable format.

        Args:
            mode (str): The mode in which to render. Common modes:
                - 'human': show rendering in a window/on-screen (if applicable)
                - 'rgb_array': return a raw image array
                - 'ansi': return a text-based representation

        Returns:
            Depends on the mode. Often None for 'human', or an np.array for 'rgb_array'.
        """
        ...

    def close(self) -> None:
        """
        Perform any necessary cleanup (e.g. close viewer windows).
        """
        ...
