import sys
import os
exp_path = os.path.dirname(os.path.abspath(__file__))
print(exp_path)
project_path = os.path.abspath(os.path.join(exp_path, ".."))
sys.path.insert(0, project_path)
print(sys.path)

import src.mdp as mdp
import numpy as np
import src.machine_teaching as machine_teaching
import src.utils as utils
from src.traj_pair import TrajPair

def create_aaai19_toy_world():
    #features is a 2-d array of tuples
    num_rows = 2
    num_cols = 3
    features =[[(1, 0), (0, 1), (1, 0)],
            [(1, 0), (1, 0), (1, 0)]]
    weights = [-1,-4]
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols)] #states indexed by row and then column
    #print(initials)
    terminals = [(0,0)]
    gamma = 0.9
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world

def create_aaai19_toy_world_3features():
    #features is a 2-d array of tuples
    num_rows = 2
    num_cols = 3
    okay = (1,0,0)
    bad = (0,1,0)
    goal = (0,0,1)
        
    features =[[goal, bad, okay],
               [okay, okay, okay]]
    weights = [-1,-4,+1]
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols)] #states indexed by row and then column
    #print(initials)
    terminals = [(0,0)]
    gamma = 0.5
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world

# world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)

def debug_mdp(world):
    print("rewards")
    world.print_rewards()

    import time

    print("values")
    t0 = time.time()
    V = mdp.value_iteration(world)
    t1 = time.time()
    world.print_map(V)
    print(t1-t0)


    print("values inplace")
    t0 = time.time()
    V = mdp.value_iteration_inplace(world)
    t1 = time.time()
    world.print_map(V)
    print(t1-t0)
    
    Q = mdp.compute_q_values(world, V)
    print("Q-values")
    print(Q)

    print("optimal policy")
    opt_policy = mdp.find_optimal_policy(world, Q=Q)
    print(opt_policy)
    print("optimal policy")
    world.print_map(world.to_arrows(opt_policy))


#debug the None for obstacles to make sure policy opt and V and Q values and reward is good.
def create_wall_3x3_world():
    num_rows = 3
    num_cols = 3
    r = (1,0)
    w = (0,1)
    features = [[w,w,w],
                [w,None,w],
                [w,r,w]]
    weights = [-10,-1]
    initials = [(2,0)]
    terminals = [(2,2)]
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals)
    #debug_mdp(world)
    return world


def create_multiple_optimal_action_mdp():
#debug the multiple optimal actions

    r = (1,0,0)
    w = (0,1,0)
    b = (0,0,1)
    features = [[w,w,w],
                [b,b,w],
                [w,r,w]]
    weights = [-10,-1,-1]
    initials = [(0,0)]
    terminals = [(2,2)]
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals)
    return world


def create_random_10x10_2feature():
    #debug value function inline with a bigger MDP
    #weird that the inplace seems slightly slower despite avoiding repeated copying of dictionary...seems to give same result though
    num_rows = 10
    num_cols = 10
    r = (1,0)
    w = (0,1)
    features = [[w if np.random.rand() < 0.5 else r for _ in range(num_cols)] for i in range(num_rows)]
    weights = [-10,-1]
    initials = [(num_rows-1,num_cols-1)]
    terminals = [(0,0)]
    gamma = 0.999
    #Only requires 2-questions to teach (2-dim feature space)
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world
    # debug_mdp(world)


def create_random_10x10_3feature():
    np.random.seed(0)
    #debug value function inline with a bigger MDP
    #weird that the inplace seems slightly slower despite avoiding repeated copying of dictionary...seems to give same result though
    num_rows = 10
    num_cols = 10
    r = (1,0,0)
    w = (0,1,0)
    b = (0,0,1)
    features = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if np.random.rand() < 0.33:
                f = r
            elif np.random.rand() < 0.66:
                f = w
            else:
                f = b
            row.append(f)
        features.append(row)
    #features = [[w if np.random.rand() < 0.5 else r for _ in range(num_cols)] for i in range(num_rows)]
    weights = -np.random.rand(3)
    initials = [(num_rows-1,num_cols-1)]
    terminals = []
    gamma = 0.99
    #Only requires 2-questions to teach (2-dim feature space)
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world
    # debug_mdp(world)



#more features in larger net
# num_rows = 50
# num_cols = 50
# r = (1,0)
# w = (0,1)
# features = [[w if np.random.rand() < 0.5 else r for _ in range(num_cols)] for i in range(num_rows)]
# weights = [-10,-1]
# initials = [(num_rows-1,num_cols-1)]
# terminals = [(0,0)]
# gamma = 0.999
#Only requires 2-questions to teach


def create_3_feature_world():
    num_rows = 3
    num_cols = 3
    # r = (1,0)
    # w = (0,1)
    r = (1,0,0)
    w = (0,1,0)
    b = (0,0,1)
    features = [[w,w,w],   #only gives one halfspace
                [b,b,w],
                [r,r,w]]
    # features = [[w,w,w],   #still buggy!
    #             [r,w,w],
    #             [r,r,w]]

                #r  #w #b  #g
    weights = [-10,-1,-3]
    initials = [(0,0)]
    terminals = [(2,2)]
    gamma = 0.9
    return mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)

def create_multi_feature_world():
    num_rows = 3
    num_cols = 3
    # r = (1,0)
    # w = (0,1)
    r = (1,0,0,0)
    w = (0,1,0,0)
    b = (0,0,1,0)
    g = (0,0,0,1)
    features = [[w,w,w],   #only gives one halfspace
                [b,b,w],
                [r,r,g]]
    # features = [[w,w,w],   #still buggy!
    #             [r,w,w],
    #             [r,r,w]]

                #r  #w #b  #g
    weights = [-10,-1,-2,+1]
    initials = [(0,0)]
    terminals = [(2,2)]
    gamma = 0.9
    return mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
#requires 18 questions to teach


def debug_demonstrations():

    world = create_random_10x10_3feature()


    print("rewards")
    world.print_rewards()

    import time

    print("features")
    utils.display_onehot_state_features(world)

    
    
    Q = mdp.compute_q_values(world)
    #print("Q-values")
    #print(Q)

    print("optimal policy")
    opt_policy = mdp.find_optimal_policy(world, Q=Q)
    #print(opt_policy)
    print("optimal policy")
    world.print_map(world.to_arrows(opt_policy))

    print(world.terminals)
    print("demo 1")
    demoA = utils.optimal_rollout_from_Qvals((1,1), 3, Q, world, 0.0001)
    for (s,a) in demoA:
        print("({},{})".format(s,world.to_arrow(a)))
    print(mdp.calculate_trajectory_feature_counts(demoA, world))

    print()
    print("demo 2")
    demoB = utils.sa_optimal_rollout_from_Qvals((1,1), (0,1), 3, Q, world, 0.0001)
    for (s,a) in demoB:
        print("({},{})".format(s,world.to_arrow(a)))
    print(mdp.calculate_trajectory_feature_counts(demoB, world))
    
    tpair = TrajPair(demoA, demoB, world, 0.0001)
    print(world.weights)


    #HMM. optimal demos can go for a long time, do we need them to last for infinite horizon??


debug_demonstrations()
# print("rewards")
# world.print_rewards()
# V = mdp.value_iteration(world)
# Q = mdp.compute_q_values(world, V)
# print("values inplace")
# world.print_map(V)

# opt_policy = mdp.find_optimal_policy(world, Q=Q)
# print("optimal policy")
# world.print_map(world.to_arrows(opt_policy))
# fcounts = mdp.calculate_expected_feature_counts(opt_policy, world)
# print("expected feature counts")
# for s in fcounts:
#     print(s)
#     print(fcounts[s])

#debug: find all half-space constraints for all possible rankings among actions, all Q(s,a) >= Q(s,b) for all a and b
# num_rows = 2
# num_cols = 3
# gray = (0,1)
# white = (1,0)
# features = [[white,gray,white],
#             [white,white,white]]
# weights = [-1,-4]
# initials = [(0,1),(0,2),(1,0),(1,1),(1,2)]
# terminals = [(0,0)]
# gamma = 0.9



#world = create_wall_3x3_world()
#world = create_aaai19_toy_world()
#world = create_aaai19_toy_world_3features()
#world = create_3_feature_world()
#world = create_random_10x10_2feature()
#world = create_random_10x10_3feature()


#TODO: test out multiple options
# world = create_multiple_optimal_action_mdp()
# teacher = machine_teaching.RankingTeacher(world, debug=False)
# ##run this method if you want optimal for teaching BEC(\pi*) without boundary conditions.
# #teacher.get_optimal_value_alignment_tests()

# ##If we want a refined ranking BEC then we can run the following
# teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)


#done: use an LP solver cvx? to remove redundancies, hopefully won't have numerical issues I used to have...

#don't need to to get test: Run set cover algorithm




#Question: Do we consider questions where the answer could be equall preference? I think for grid worlds we should! but this can be done as a postprocessing step 
#for now since we just need to brute force it. Anyways I think searching over the optimal machine testing sets should be pretty quick to detect equal
#preference questions and we can collapse A > B and B > A into one question A ? B. I guess we'll collapse anyways but we can recognize overlaps if we see then
# and make sure to score the agent correctly on equal preference answers.

#TODO: debug when there is an equality preference. The above example doesn't seem right... why only one non-redundant constraint??!!

#TODO: think of some experiments to run

#TODO: figure out how we should figure out optimal testing sets...do we want trajectories where we give a query at some of the states?

#TODO: get policy evaluation working for stochastic and deterministic policies under any MDP

#TODO: do we really want preferences? This I think will be smaller than BEC(\pi^*), no? Is that a problem? we'll find out... there will be more halfplanes for sure...

