import numpy as np
import gymnasium as gym

env = gym.make("MountainCar-v0")
env.reset()

class MounTainCarAgent:
    def __init__(
        self,
        c_learning_rate: float,
        c_discount_value: float,
        v_epsilon: float,
        c_start_ep_epsilon_decay: float,
        c_no_of_eps: int,
        q_table_size: tuple[int, int],
        env: gym.Env,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.lr = c_learning_rate
        self.discount_value = c_discount_value

        self.epsilon = v_epsilon
        self.c_no_of_eps = c_no_of_eps
        self.c_start_ep_epsilon_decay = c_start_ep_epsilon_decay
        self.c_end_ep_epsilon_decay = self.c_no_of_eps // 2
        self.epsilon_decay = self.epsilon / (self.c_end_ep_epsilon_decay - self.c_start_ep_epsilon_decay)
        
        self.q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size
        self.q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))
        
    def convert_state(self, obs: tuple[float, float]) -> tuple[int, int]:
        """Converts a continuous state into a discrete state."""
        q_state = (obs - env.observation_space.low) // self.q_table_segment_size
        return tuple(q_state.astype(int))

    def get_action(self, obs: tuple[float, float]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.randint(0, env.action_space.n)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return np.argmax(self.q_table[obs])

    def update(
        self,
        current_obs: tuple[float, float],
        action: int,
        reward: float,
        next_real_obs: tuple[float, float],
    ):
        """Updates the Q-value of an action."""
        
        next_obs = self.convert_state(next_real_obs)
        current_q_value = self.q_table[current_obs + (action,)]
        new_q_value = (1 - self.lr) * current_q_value + self.lr * (reward + self.discount_value * np.max(self.q_table[next_obs]))
        self.q_table[current_obs + (action,)] = new_q_value

    def decay_epsilon(self, ep: int):
        if self.c_end_ep_epsilon_decay >= ep > self.c_start_ep_epsilon_decay:
            self.epsilon -= self.epsilon_decay
        # print(f"epsilon: {self.epsilon}")