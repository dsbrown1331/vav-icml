#I want to be able to run this without the command line
#going to stick with dqn and lunar lander for now

from featurizer import indicator_feature
import numpy as np

import os
import sys
import argparse
import importlib
import warnings

sys.path.insert(0,'rl-baselines-zoo/')
print("path", sys.path)


# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import utils.import_envs  # pytype: disable=import-error
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from utils.utils import StoreDict

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def rollout_halfspaces(env_id='CartPole-v1',algo='dqn',num_samples=20, precision=0.0001, render=False):
    seed = 0
    folder = 'rl-baselines-zoo/trained_agents'
    n_envs = 1
    no_render = False
    deterministic = True
    stochastic = False
    norm_reward=False

    
    log_path = os.path.join(folder, algo)


    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=False)

    
    set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    log_dir = None

    env_kwargs = {}

    
    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=not no_render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    env = gym.make('CartPole-v1')
    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    state = None

    embedder = indicator_feature
    halfspaces = {}
    
    for i in range(num_samples):
        print("+"*10)
        #sample random state to start in
        
        #TODO: maybe reset with random actions? How to make it realistic? Does it matter. Let's just try random for now to test weird edge cases.
        obs = env.reset(uniform=True) #sample more uniformly than typical
        start_state = obs.copy()
        print("start state", obs)

        #find out the "near optimal" action for this state to compare other actions to
        opt_action, _ = model.predict(obs, state=state, deterministic=deterministic)
        #take this action
        print("TEACHER ACTION", opt_action)
        obs = env.reset(start_state=start_state)
        print("init state", obs)
        if render:
            env.render()
        # input()
        ep_ret = 0
        fcounts = embedder(start_state)
        #do initial action
        obs, r, done, info = env.step(opt_action) # take a random action

        fcounts += embedder(obs)  #TODO: discount??
        ep_ret += r
        #print(r, obs)
        if done:
            #sample again, since started with terminal state
            continue



        #run tester policy thereafter
        while True:

            #env.render()

            #TODO: sample within allowable range of angle and position
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            # Random Agent
            # action = [env.action_space.sample()]
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            #a = env.action_space.sample()
            #print(obs, action)
            obs, r, done, info = env.step(action) # take a random action
            
            fcounts += embedder(obs)
            #print(obs)
            #print(done)
            ep_ret += r
            #print(r, obs)
            if done:
                print("final state", obs)
                print("return", ep_ret)
                print("fcounts", fcounts)
                opt_fcounts = fcounts
                break



        # input()
        #obs = env.reset_state(env.observation_space.sample())

        #rollout once for each action and compute feature counts
        
        
        fcount_vectors = []
        init_actions = []
        ##rollout code:
        for init_action in range(env.action_space.n):
            if init_action == opt_action:
                #don't need to roll this out since we already did
                continue
            print("ACTION", init_action)
            obs = env.reset(start_state=start_state)
            print("init state", obs)
            if render:
                env.render()
            # input()
            ep_ret = 0
            fcounts = embedder(start_state)
            #do initial action
            obs, r, done, info = env.step(init_action) # take a random action

            fcounts += embedder(obs)  #TODO: discount??
            ep_ret += r
            #print(r, obs)
            if done:
                print("final state", obs)
                print("return", ep_ret)
                print("fcounts", fcounts)
                fcount_vectors.append(fcounts)
                init_actions.append(init_action)
                continue



            #run tester policy thereafter
            while True:

                #env.render()

                #TODO: sample within allowable range of angle and position
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                # Random Agent
                # action = [env.action_space.sample()]
                # Clip Action to avoid out of bound errors
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                #a = env.action_space.sample()
                #print(obs, action)
                obs, r, done, info = env.step(action) # take a random action
                
                fcounts += embedder(obs)
                #print(obs)
                #print(done)
                ep_ret += r
                #print(r, obs)
                if done:
                    print("final state", obs)
                    print("return", ep_ret)
                    print("fcounts", fcounts)
                    break

            normal_vector = opt_fcounts - fcounts
            print("action {} over {} => fcount diff = {}".format(opt_fcounts, init_action, normal_vector))
            if np.linalg.norm(normal_vector) > precision:
                halfspaces[tuple(start_state), init_action, opt_action] = normal_vector
        input()
        #TODO: put this inside one of the value alignment verification classes to get sa_fcount_diffs and hopefully reuse that code
        #then visualize test cases

        # input()
    # for _ in range(args.n_timesteps):
    #     action, state = model.predict(obs, state=state, deterministic=deterministic)
    #     # Random Agent
    #     # action = [env.action_space.sample()]
    #     # Clip Action to avoid out of bound errors
    #     if isinstance(env.action_space, gym.spaces.Box):
    #         action = np.clip(action, env.action_space.low, env.action_space.high)
    #     obs, reward, done, infos = env.step(action)
    #     if not args.no_render:
    #         env.render('human')

    #     episode_reward += reward
    #     ep_len += 1

    #     if args.n_envs == 1:
    #         # For atari the return reward is not the atari score
    #         # so we have to get it from the infos dict
    #         if is_atari and infos is not None and args.verbose >= 1:
    #             episode_infos = infos.get('episode')
    #             if episode_infos is not None:
    #                 print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
    #                 print("Atari Episode Length", episode_infos['l'])

    #         if done and not is_atari and args.verbose > 0:
    #             # NOTE: for env using VecNormalize, the mean reward
    #             # is a normalized reward when `--norm_reward` flag is passed
    #             print("Episode Reward: {:.2f}".format(episode_reward))
    #             print("Episode Length", ep_len)
    #             state = None
    #             episode_rewards.append(episode_reward)
    #             episode_lengths.append(ep_len)
    #             episode_reward = 0.0
    #             ep_len = 0

    #         # Reset also when the goal is achieved when using HER
    #         if done or infos.get('is_success', False):
    #             if args.algo == 'her' and args.verbose > 1:
    #                 print("Success?", infos[0].get('is_success', False))
    #             # Alternatively, you can add a check to wait for the end of the episode
    #             # if done:
    #             obs = env.reset()
    #             if args.algo == 'her':
    #                 successes.append(infos[0].get('is_success', False))
    #                 episode_reward, ep_len = 0.0, 0

    # if args.verbose > 0 and len(successes) > 0:
    #     print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    # if args.verbose > 0 and len(episode_rewards) > 0:
    #     print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    # if args.verbose > 0 and len(episode_lengths) > 0:
    #     print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not no_render:
        if n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()

    return halfspaces


if __name__ == '__main__':
    env_id='CartPole-v1'
    algo='ppo2'
    num_samples=30
    sa_halfspaces = rollout_halfspaces(env_id, algo, num_samples, render=True)
    for entry in sa_halfspaces:
        print(entry, sa_halfspaces[entry])
