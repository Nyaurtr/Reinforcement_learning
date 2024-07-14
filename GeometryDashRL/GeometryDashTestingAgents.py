import GeometryDashEnvironments
from GeometryDashAgent import DQNAgent
import numpy as np
import cv2  # OpenCV for rendering

def test_agent(env, agent, episodes):
    agent.load("./GeometryDashRL/geometry_dash_dqn.weights.h5")

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, *state.shape])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, epsilon=0.01)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, *next_state.shape])
            state = next_state
            total_reward += reward

            # Render the game
            env.render()

        print(f"Episode: {e}/{episodes}, Score: {total_reward}")

    # Destroy all the windows when done testing
    cv2.destroyAllWindows()

if __name__ == "__main__":
    env = GeometryDashEnvironments.GeometryDashEnv()
    state_shape = (84, 84, 3)
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)
    test_agent(env, agent, episodes=10)
