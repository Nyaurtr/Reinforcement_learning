import gymnasium as gym
from Model import MounTainCarAgent
import numpy as np
from tqdm import tqdm

env = gym.make("MountainCar-v0")
env.reset()

def training(max_ep_reward, max_ep_action_list, max_start_state, c_no_of_eps, env, agent):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=c_no_of_eps)
    for ep in tqdm(range(c_no_of_eps)):
        ep_reward = 0
        current_obs = agent.convert_state(env.reset()[0])
        done = False
        action_list = []
        step = 0

        while not done and step < 200:
            action = agent.get_action(current_obs)
            action_list.append(action)
            next_real_obs, reward, done, _, _ = env.step(action=action)
            ep_reward += reward

            # print(next_real_obs)

            if done:
                if next_real_obs[0] >= env.unwrapped.goal_position:
                    if ep_reward > max_ep_reward:
                        max_ep_reward = ep_reward
                        max_ep_action_list = action_list
                        max_start_state = current_obs
            else:
                next_obs = agent.convert_state(next_real_obs)
                agent.update(current_obs, action, reward, next_real_obs)
                current_obs = next_obs
                
            
            step += 1
        
        agent.decay_epsilon(ep=ep)
    
    return max_ep_reward, max_ep_action_list, max_start_state

if __name__ == "__main__":
    # hyperparameters
    c_learning_rate = 0.1
    c_discount_value = 0.9
    c_no_of_eps = 10000
    c_show_each = 1000

    v_epsilon = 0.9
    c_start_ep_epsilon_decay = 1
    q_table_size = [20, 20]

    agent = MounTainCarAgent(
        c_discount_value=c_discount_value,
        c_learning_rate=c_learning_rate,
        v_epsilon=v_epsilon,
        c_start_ep_epsilon_decay=c_start_ep_epsilon_decay,
        c_no_of_eps=c_no_of_eps,
        q_table_size=q_table_size,
        env=env,
    )
    
    max_ep_reward = -np.inf
    max_ep_action_list = []
    max_start_state = None
    
    result = training(max_ep_reward = max_ep_reward, max_ep_action_list = max_ep_action_list, max_start_state = max_start_state, c_no_of_eps = c_no_of_eps, env = env, agent = agent)
    
    with open("mountain_car_game_action.txt", "w") as file:
        file.write(str(result[2]))
        for i in result[1]:
            file.write("\n")
            file.write(str(i))
    print("Code has been exported to 'mountain_car_game.txt' successfully!")