from typing import Protocol, Tuple, Any

"""
The main Gymnasium class for implementing Reinforcement Learning Agents environments.

The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the step() and reset() functions. An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

The main API methods that users of this class need to know are:

    step() - Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    reset() - Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    render() - Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text.
    close() - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

Environments have additional attributes for users to understand the implementation:
    action_space - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    observation_space - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    spec - An environment spec that contains the information used to initialize the environment from gymnasium.make()
    metadata - The metadata of the environment, e.g. {“render_modes”: [“rgb_array”, “human”], “render_fps”: 30}. For Jax or Torch, this can be indicated to users with “jax”=True or “torch”=True.
    np_random - The random number generator for the environment. This is automatically assigned during super().reset(seed=seed) and when assessing np_random.
"""


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

    @property
    def action_space(self) -> Any:
        """
        The action space of the environment.
        """
        ...

    @property
    def observation_space(self) -> Any:
        """
        The observation space of the environment.
        """
        ...

    @property
    def set_random(self, seed: int) -> None:
        """
        Set the random seed for the environment.
        """
        ...

    def step(self, action: int) -> Tuple[Any, float, bool]:
        """
        Run one timestep of the environment’s dynamics.

        Args:
            action (int): The action taken by the agent.

        Returns:
            observation (Any): The agent’s observation of the current environment.
            reward (float): The reward for the taken action.
            done (bool): Whether the episode has ended.

        Note:
            In a Gymnasium environment, this function often returns an additional
            'info' dictionary. You can include that if desired.
        """
        ...

    def reset(self) -> Any:
        """
        Reset the environment to its initial state.

        Returns:
            observation (Any): The initial observation for the agent.
        """
        ...

    def render(self, mode: str = "human") -> Any:
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
