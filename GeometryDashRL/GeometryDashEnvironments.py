import gym
from gym import spaces
import numpy as np
import cv2  # OpenCV for image processing
import random

class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: no action, 1: jump
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        # Game state variables
        self.state = None
        self.score = 0
        self.done = False
        self.player_y = 50  # Player's vertical position
        self.velocity_y = 0  # Player's vertical velocity
        self.gravity = 1
        self.jump_strength = -10
        self.obstacles = []
        self.obstacle_speed = -5

    def reset(self):
        self.state = self._get_initial_state()
        self.score = 0
        self.done = False
        self.player_y = 50
        self.velocity_y = 0
        self.obstacles = self._generate_obstacles()
        return self.state

    def step(self, action):
        if action == 1:  # Jump
            self.velocity_y = self.jump_strength

        # Apply gravity
        self.velocity_y += self.gravity
        self.player_y += self.velocity_y

        # Move obstacles
        for obstacle in self.obstacles:
            obstacle[0] += self.obstacle_speed

        # Check for collisions and score
        self.done = self._check_done()
        reward = self._compute_reward()

        # Update state
        self.state = self._get_next_state()
        
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        if mode == 'human':
            img = self.state
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Geometry Dash', img)
            cv2.waitKey(1)

    def _get_initial_state(self):
        state = np.zeros((84, 84, 3), dtype=np.uint8)
        state = self._update_state(state)
        return state

    def _apply_action(self, action):
        if action == 1:  # Jump
            self.velocity_y = self.jump_strength

    def _get_next_state(self):
        state = np.zeros((84, 84, 3), dtype=np.uint8)
        state = self._update_state(state)
        return state

    def _compute_reward(self):
        if self.done:
            return -100
        else:
            return 1

    def _check_done(self):
        if self.player_y < 0 or self.player_y > 83:
            return True

        for obstacle in self.obstacles:
            if obstacle[0] < 5 and obstacle[0] > -5:  # Simple collision detection
                if self.player_y > obstacle[1]:
                    return True

        return False

    def _update_state(self, state):
        # Draw the player
        state[int(self.player_y):int(self.player_y)+5, 5:10] = [255, 255, 255]

        # Draw the obstacles
        for obstacle in self.obstacles:
            state[int(obstacle[1]):int(obstacle[1])+10, int(obstacle[0]):int(obstacle[0])+5] = [255, 0, 0]

        return state

    def _generate_obstacles(self):
        obstacles = []
        for i in range(5):
            x = random.randint(84, 150) + i * 20
            y = random.randint(50, 70)
            obstacles.append([x, y])
        return obstacles

    def _render_game(self):
        self.render()