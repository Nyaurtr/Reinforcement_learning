import gymnasium as gym

# Read the content of the text file into a list
with open("mountain_car_game_action.txt", "r") as file:
    lines = file.readlines()
    
# Alternatively, if you want to store the content in a list
# without newline characters, you can use list comprehension

# Convert the first two lines to integers and create a tuple
max_start_state = (int(lines[0]), int(lines[1]))

# Convert the rest of the lines to integers and create a list
max_ep_action_list = [int(line) for line in lines[2:]]


env = gym.make("MountainCar-v0", render_mode='human')
env.reset()
env.state = max_start_state
for action in max_ep_action_list:
    env.step(action)
    env.render()

done = False
while not done:
    _, _, done,_, _ = env.step(0)
    env.render()

env.close()