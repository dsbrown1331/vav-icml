import gym
import copy
env = gym.make('LunarLander-v2')
env.seed(111) # we cna fix the background for now
env.action_space.np_random.seed(123) #fix random actions for now
obs = env.reset()
while True:

    env.render()
    a = env.action_space.sample()
    print(obs, a)
    obs, r, done, info = env.step(env.action_space.sample()) # take a random action
    print(r, obs)
    if done:
        break
