import numpy as np
import src.mdp
import random
import src.machine_teaching

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

    Q = mdp.compute_q_values(world, V)

    opt_policy = mdp.find_optimal_policy(world, Q=Q)
    print("optimal policy")
    world.print_map(world.to_arrows(opt_policy))

def create_random_features_row_col_m(num_rows, num_cols, num_features):
    assert(num_features <= num_rows * num_cols) #we want at least one of each feature
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]
    #print(features)
    #state_features = [[random.choice(features) for _ in range(num_cols)] for _ in range(num_rows)]

    #make sure that there is at least one of each feature so we can have different policies.
    state_coords = [(r,c) for r in range(num_rows) for c in range(num_cols)]
    #sample without replacement num_feature states so we have one of each feature
    state_sample = random.sample(state_coords, num_features)
    cnt = 0
    state_features = []
    for r in range(num_rows):
        row_features = []
        for c in range(num_cols):
            if (r,c) not in state_sample:
                #just select at random
                row_features.append(random.choice(features))
            else:
                #iterate through each feature to make sure it appears at least once
                row_features.append(features[cnt])
                cnt += 1
        state_features.append(row_features)
    

    #print(state_features)
    return state_features

def create_row_x_col_m_feature_mdp(rows, cols, num_features):
    #No terminal state for now and positive and negative reward
    #debug value function inline with a bigger MDP
    #weird that the inplace seems slightly slower despite avoiding repeated copying of dictionary...seems to give same result though
    num_rows = rows
    num_cols = cols
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]
    #print(features)
    state_features = [[random.choice(features) for _ in range(num_cols)] for _ in range(num_rows)]
    #print(state_features)
    
    weights = 1.0 - 2.0 * np.random.rand(num_features)
    initials = [(num_rows // 2, num_cols // 2)]
    terminals = []#[(num_rows-1,num_cols-1)]
    gamma = 0.95
    #Only requires 2-questions to teach (2-dim feature space)
    world = mdp.LinearFeatureGridWorld(state_features, weights, initials, terminals, gamma)
    #debug_mdp(world)
    return world



if __name__ == "__main__":
    world = create_row_x_col_m_feature_mdp(5,5,5)
    teacher = machine_teaching.RankingTeacher(world, debug=False)
    teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)
