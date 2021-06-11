# Let's try Anca's idea of having a test specifically designed for querying actions.

#for now I'm going to just randomly sample to get policies and add the zero vector too.
#I think there should be a way to sample vectors along each halfspace constraint too
# Just need to make sure they are orthogonal to halfspace normal vector. There could be a lot of these too
# since I have n-1 degrees of freedom, but I could sample randomly from each hyperplane and remove duplicates.
# not going to do that for now...

#first sample random reward functions and print out the unique policies:
import sys
import os
exp_path = os.path.dirname(os.path.abspath(__file__))
print(exp_path)
project_path = os.path.abspath(os.path.join(exp_path, ".."))
sys.path.insert(0, project_path)
print(sys.path)
import src.grid_worlds as grid_worlds
import src.mdp as mdp
import copy
import numpy as np
import random
import src.machine_teaching as machine_teaching
import data_analysis.mdp_family_teaching.twoFeatureFeasibleRegionPlot as plot_aec
import data_analysis.plot_grid as mdp_plot
import src.utils as utils


def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

def get_perpendiculars(normal_vec):
    #only works for 2-d return both directions since there are two possibile directions to be perpendicular
    if normal_vec[0] == 0 and normal_vec[1] == 0:
        return [np.array([0,0])]
    elif normal_vec[0] == 0:
        return [np.array([1,0]), np.array([-1,0])]
    elif normal_vec[1] == 0:
        return [np.array([0,1]), np.array([0,-1])]
    else:
        return [np.array([1, -normal_vec[0]/ normal_vec[1]]),np.array([-1,normal_vec[0]/ normal_vec[1]])] 
    

#TODO: make this more general for any epsilon, currently optimized for eps = 0
def generate_test_action_query(true_world, num_reward_samples, horizon, num_rollouts, epsilon, delta, seed, precision, debug):

    rollout_length = horizon
    #calculate the optimal policy for true_world
    state_features = true_world.features
    V = mdp.value_iteration(true_world, epsilon=precision)
    true_exp_return = np.mean([V[s] for s in true_world.initials])
    Qopt = mdp.compute_q_values(true_world, V=V, eps=precision)
    opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt, epsilon=precision)


    

    num_features = true_world.get_num_features()
    init_seed = seed
    np.random.seed(init_seed)
    random.seed(init_seed)

    num_eval_policies_tries = num_reward_samples

    eval_policies = []
    eval_Qvalues = []
    eval_weights = []
    eval_policy_losses = []
    num_eval_policies = 0
    for i in range(num_eval_policies_tries):
        rand_world = copy.deepcopy(true_world)
        #print("trying", i)
        #change the reward weights
        eval_weight_vector = random_weights(num_features)
        rand_world.weights = eval_weight_vector
        #find the optimal policy under this MDP
        Qval = mdp.compute_q_values(rand_world, eps=precision)
        eval_policy = mdp.find_optimal_policy(rand_world, Q=Qval, epsilon=precision)
        #only save if not equal to optimal policy
        if eval_policy  not in eval_policies and eval_policy != opt_policy:
            if debug:
                print("found distinct eval policy")
                print("weights", eval_weight_vector)

                rand_world.print_map(rand_world.to_arrows(eval_policy))
        
            eval_exp_values = mdp.policy_evaluation(eval_policy, true_world, epsilon=precision)
            eval_exp_return = np.mean([eval_exp_values[s] for s in true_world.initials])
            eval_policy_loss = true_exp_return - eval_exp_return
            
            #check if value aligned for all states
            epsilon_aligned = True
            for s in true_world.states:
                if V[s] - eval_exp_values[s] > epsilon:
                    epsilon_aligned = False
                    break
            
            if not epsilon_aligned:
                #print("true exp return", true_exp_return, "eval exp return", eval_exp_return)
                eval_policies.append(eval_policy)
                eval_Qvalues.append(Qval)
                eval_weights.append(eval_weight_vector)
                
                eval_policy_losses.append(eval_policy_loss)
                num_eval_policies += 1

    print("There are {} distinct optimal policies not including teacher's policy when sampling randomly".format(len(eval_policies)))



    #add the zero reward #still not sure if this should combine with the all 1's vector or not....
    eval_weight_vector = np.zeros(true_world.get_num_features())
    rand_world = copy.deepcopy(true_world)
    #print("trying", i)
    #change the reward weights
    rand_world.weights = eval_weight_vector
    #find the optimal policy under this MDP
    Qval = mdp.compute_q_values(rand_world, eps=precision)
    eval_policy = mdp.find_optimal_policy(rand_world, Q=Qval, epsilon=precision)
    #only save if not equal to optimal policy
    if eval_policy  not in eval_policies:
        if debug:
            print("found distinct eval policy")
            print("weights", eval_weight_vector)

            rand_world.print_map(rand_world.to_arrows(eval_policy))

        eval_exp_values = mdp.policy_evaluation(eval_policy, true_world, epsilon=precision)
        eval_exp_return = np.mean([eval_exp_values[s] for s in true_world.initials])
        eval_policy_loss = true_exp_return - eval_exp_return
        
        #check if value aligned for all states
        epsilon_aligned = True
        for s in true_world.states:
            if V[s] - eval_exp_values[s] > epsilon:
                epsilon_aligned = False
                break
        
        if not epsilon_aligned:
            #print("true exp return", true_exp_return, "eval exp return", eval_exp_return)
            eval_policies.append(eval_policy)
            eval_Qvalues.append(Qval)
            eval_weights.append(eval_weight_vector)
            
            eval_policy_losses.append(eval_policy_loss)
            num_eval_policies += 1

    print("There are {} distinct non eps aligned optimal policies not including teacher's policy and including trivial policy".format(len(eval_policies)))
    for i,p in enumerate(eval_policies):
        print(i)
        print(eval_weights[i])
        print(eval_policy_losses[i])
        rand_world.print_map(true_world.to_arrows(p))



    #okay, let's assume that I have an epsilon and delta for testing now.


    state_list = list(true_world.states)
    num_states = len(state_list)
    num_non_aligned = len(eval_policies)  #TODO: check through these first to see if they are eps-aligned.
    #fill this in with on policy rollouts per eval policy
    robot_state_detected_probs = np.zeros((num_states, num_non_aligned))

    for s_idx, s_init in enumerate(state_list):
        #print(s_init)
        for i,p in enumerate(eval_policies):
            # print("eval policy", i)
            # print(eval_weights[i])
            # print(eval_policy_losses[i])
            #rand_world.print_map(true_world.to_arrows(p))
            #def generate_candidate_trajectories(self):
            trajs = []
            for k in range(num_rollouts):
                detected = False
                traj = utils.optimal_rollout_from_Qvals(s_init, rollout_length, eval_Qvalues[i], true_world, precision)
                #print(traj)
                #check if it takes a suboptimal action #TODO make this general for eps
                for s,a in traj:
                    #get an optimal action in this state
                    if type(opt_policy[s]) is list:
                        opt_action = opt_policy[s][0]
                    else:
                        opt_action = opt_policy[s]
                    #print("opt action", opt_action)
                    if Qopt[s, opt_action] - Qopt[s,a] > epsilon:
                        #took an action that isn't eps aligned!
                        detected = True
                        break
                if detected:
                    robot_state_detected_probs[s_idx, i] += 1

    robot_state_detected_probs /= num_rollouts

    if debug:
        for s_idx, s in enumerate(true_world.states):
            print("state", s)
            print(robot_state_detected_probs[s_idx])


    #okay so now that we have the detected counts, we need to run breadth first search
    #page 82 R&N

    #start with all single state rollouts and check if they satisfy delta, if none do, then add to the frontier and start search
    def goal_test_delta(state_set, delta):
        #check whether each agent goes undetected with no more than prob delta
        #vectorized version
        prob_dont_detect_ind_states = 1 - robot_state_detected_probs
        prob_dont_detect_test = np.ones(num_non_aligned) #each eval policy starts with prob 1 of not being detected (with no test)
        for s_idx in state_set:
            prob_dont_detect_test *= prob_dont_detect_ind_states[s_idx]
        #check if all eval policies < delta
        if (prob_dont_detect_test < delta).all():
            return True
        else:
            return False
                    #  0        1       2       3      4        5
    print(state_list) #(0, 1), (1, 2), (0, 0), (1, 1), (0, 2), (1, 0)]
    goal_test_delta({4, 1, 3}, delta)


    def breadth_first_search(state_list):
        import queue
        frontier = queue.Queue() #FIFO
        explored = set()

        #check singleton states
        solution = set()
        solution_found = False
        for s_idx, s in enumerate(state_list):
            print(s)
            if s == (0,0):
                print("debug")
            if goal_test_delta({s_idx}, delta): #check if this state works for all policies 
                solution = {s_idx}
                solution_found = True
                break
            else: 
                frontier.put({s_idx})
                explored.add(frozenset({s_idx}))
        #start full depth first search by adding more and more states to the test
        if not solution_found:
            while frontier.qsize() > 0:
                #breadth first search
                test = set(frontier.get())
                if debug:
                    print("expanding test", test)
                for s_idx, s in enumerate(state_list):
                    #add new states to test, check if they are solutions and add to frontier
                    new_test = copy.deepcopy(test)
                    new_test.add(s_idx)
                    #print(new_test)
                    if new_test not in explored:
                        if debug:
                            print("testing new test", new_test)
                        if goal_test_delta(new_test, delta): #check if this state works for all policies 
                            solution = new_test
                            solution_found = True
                            if debug:
                                print("solution found")
                            return solution
                        else:
                            if debug:
                                print("adding to frontier and explored")
                            frontier.put(new_test)
                            explored.add(frozenset(new_test))
        return None

    solution = breadth_first_search(state_list)
    if solution:
        print("solution", [state_list[i] for i in solution])

        #print out the probability of not failing each agent
        prob_dont_detect_ind_states = 1 - robot_state_detected_probs
        prob_dont_detect_test = np.ones(num_non_aligned) #each eval policy starts with prob 1 of not being detected (with no test)
        for s_idx in solution:
            prob_dont_detect_test *= prob_dont_detect_ind_states[s_idx]
        print(prob_dont_detect_test)


        for i,p in enumerate(eval_policies):
            print(i)
            print(eval_weights[i])
            print(eval_policy_losses[i])
            rand_world.print_map(true_world.to_arrows(p))
    if solution:
        return [state_list[i] for i in solution]
    else:
        return None #no solution found

if __name__=="__main__":
    precision = 0.00001
    seed = 1234
    true_world = grid_worlds.create_aaai19_toy_world()
    num_reward_samples = 100
    epsilon = 1.0 #TODO make this more general for any epsilon
    delta = 0.05
    num_rollouts = 100
    horizon = 1
    debug = False
    soln = generate_test_action_query(true_world, num_reward_samples, horizon, num_rollouts, epsilon, delta, seed, precision, debug )
    print("solution", soln)