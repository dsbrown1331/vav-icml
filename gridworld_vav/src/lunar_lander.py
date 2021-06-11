import gym
import copy
env = gym.make('LunarLander-v2')
env.seed(111) # we cna fix the background for now
env.action_space.np_random.seed(123) #fix random actions for now
env.reset()
for step in range(60):
    #input()
    env.render()
    #save info before action
    if step == 55:
        save_state = copy.copy(info)
        print("save pos", save_state['posx'], save_state['posy'])
        input("Let's restart here")
    elif step > 55:
        print(step, info['posx'], info['posy'])

    obs, r, done, info = env.step(0)#env.action_space.sample()) # take a random action

    if step == 55:
        #save after state
        after_55 = copy.copy(info)
        obs_after_55 = obs
    # print(obs)
    # print(r)
    # print(done)
    # print(obs)
    # print(info['posx'], info['posy'])
    # print()
    
#print('SAVED lander', save_state['lander'])
print("recover pos", save_state['leg_posx'], save_state['leg_posy'])
obs, r, done, info = env.reset(game_state = save_state, action = 0)
#print('SAVED lander',  save_state['lander'])
print("obs after 55")
print(obs_after_55)
print("obs after reset")
print(obs)
env.render()
print("recovered pos", after_55['leg_posx'])
print("reset state", info['leg_posx'])
input()
for step in range(56,60):
    env.render()
    print(step, info['posx'], info['posy'])
    obs, r, done, info = env.step(0)#env.action_space.sample()) # take a random action
    # print(obs)
    # print(r)
    # print(done)
    #print(obs)
    
    # print()

input()
env.close()



#Questions, how do we reset the env with a particular state?


#didn't work. I think I need to rewrite all the leg and body code with and without previous state

#hmm I tried, but still seems to not be working. Why is it different? Need to run debugger and find out.

#okay so I reset the leg linearVelocity and angular velocity


#TODO: doesn't work!! Need to debug. The posx and posy and angle for the legs is wrong for some reason... I thought I had it working...
#Need to run two debuggers and see when it changes
#check None action!

#TODO test with non-zero actions and make sure it works and see if I can get the particles to also work.
#TODO test with dispersion noise
