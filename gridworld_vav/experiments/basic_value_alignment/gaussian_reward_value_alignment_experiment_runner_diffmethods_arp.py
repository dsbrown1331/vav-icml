#I want to rerun things with the ARP rather than the AEC...

import sys
import os
exp_path = os.path.dirname(os.path.abspath(__file__))
print(exp_path)
project_path = os.path.abspath(os.path.join(exp_path, "..", ".."))
sys.path.insert(0, project_path)
print(sys.path)

import src.experiment_utils as eutils
import src.utils as utils
import src.mdp as mdp
import src.machine_teaching
import copy
import numpy as np
import src.value_alignment_verification as vav
import src.alignment_heuristics as ah
import random
import sys
import src.machine_teaching as machine_teaching

#evaluate several different verification methods and compute accuracies


def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

def sample_gaussian_weights(mean_vec, stdev_scalar):
    weights =  np.random.normal(0.0, stdev_scalar, len(mean_vec)) + mean_vec
    return weights / np.linalg.norm(weights)

init_seed = 1234
num_trials = 10  #number of mdps with random rewards to try
num_eval_policies_tries = 50

#scot params
num_rollouts = 20
#used for scot and traj comparisons
rollout_length = 20  #should be more than  np.log(eps * (1-gamma))/np.log(gamma) to gurantee epsilong accuracy

# how far to sample
sigma = 0.4

debug = False
precision = 0.00001
num_rows_list = [4,5,6,7,8]#[4,8,16]
num_cols_list = [4,5,6,7,8]#[4,8,16]
num_features_list = [3,4,5,6,7,8]
#verifier_list =['arp-pref',"arp-bb", "arp-w","scot","state-value-critical-0.2"]
verifier_list =["arp-pref"]


exp_data_dir = os.path.join(project_path, "results", "arp_gaussian")

if not os.path.exists(exp_data_dir):
    os.makedirs(exp_data_dir)

for num_features in num_features_list:
    for num_rows in num_rows_list:
        num_cols = num_rows #keep it square grid for  now

        result_writers = []
        for i, verifier_name in enumerate(verifier_list):
            filename = "arp{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            full_path = os.path.join(exp_data_dir, filename)
            print("writing to", full_path)
            result_writers.append(open(full_path,'w'))
            #input()

        for r_iter in range(num_trials):
            print("="*10, r_iter, "="*10)
            print("features", num_features, "num_rows", num_rows)
            ##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
            #MDP is deterministic with fixed number or rows, cols, and features
            #try a variable number of eval policies since bigger domains can have more possible policies (this is just a heuristic to make sure we try a lot but not as many for really small mdps)
            # 2 * num_features * num_rows * num_cols #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
            initials = [(i,j) for i in range(num_rows) for j in range(num_cols)]
            terminals = []#[(num_rows-1,num_cols-1)]
            gamma = 0.9
            seed = init_seed + r_iter 
            print("seed", seed)
            np.random.seed(seed)
            random.seed(seed)

            #First let's generate a random MDP
            state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
            #print("state features\n",state_features)
            true_weights = random_weights(num_features)
            true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
            V = mdp.value_iteration(true_world, epsilon=precision)
            Qopt = mdp.compute_q_values(true_world, V=V, eps=precision)
            opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt, epsilon=precision)
            
            if debug:
                print("true weights: ", true_weights)  
            
                print("rewards")
                true_world.print_rewards()
                print("value function")
            
                true_world.print_map(V)
                print("mdp features")
                utils.display_onehot_state_features(true_world)
            
                print("optimal policy")
                true_world.print_map(true_world.to_arrows(opt_policy))
            
            #now find a bunch of other optimal policies for the same MDP but with different weight vectors.
            world = copy.deepcopy(true_world)
            eval_policies = []
            eval_Qvalues = []
            eval_weights = []
            num_eval_policies = 0
            for i in range(num_eval_policies_tries):
                #print("trying", i)
                #change the reward weights
                
                eval_weight_vector = sample_gaussian_weights(true_weights, sigma)
                # print("true weights", true_weights)
                # print("new weights", eval_weight_vector)
                world.weights = eval_weight_vector
                #find the optimal policy under this MDP
                Qval = mdp.compute_q_values(world, eps=precision)
                eval_policy = mdp.find_optimal_policy(world, Q=Qval, epsilon=precision)
                #only save if not equal to optimal policy
                #
                if eval_policy != opt_policy:# and eval_policy not in eval_policies:
                    if debug:
                        print("found distinct eval policy")
                        world.print_map(world.to_arrows(eval_policy))
                
                    eval_policies.append(eval_policy)
                    eval_Qvalues.append(Qval)
                    eval_weights.append(eval_weight_vector)
                    num_eval_policies += 1

            print("There are {} distinct optimal policies".format(len(eval_policies)))
            if len(eval_policies) == 0:
                print("The only possible policy is the optimal policy. There must be a problem with the features. Can't do verification if only on policy possible!")
                sys.exit()
                

            print()
            print("Generating verification tests")

            #TODO: save computation by solving for halfspaces once for ARP-w and ARP-bb
            teacher = machine_teaching.StateActionRankingTeacher(true_world, Qopt, opt_policy, debug=debug, epsilon=precision)
        
            #TODO: we don't need the tests, just the halfspaces, but we do need to know which are equality
            tests, halfspaces = teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False, compare_optimal = False)



            for vindx, verifier_name in enumerate(verifier_list):
                tester = None
                size_verification_test = None

                if "state-value-critical-" in verifier_name:
                    critical_value_thresh = float(verifier_name[len("state-value-critical-"):])
                    #print("critical value", critical_value_thresh)
                    tester = ah.CriticalStateActionValueVerifier(true_world, Qopt, opt_policy, critical_value_thresh, precision=precision, debug=debug)
                
                elif verifier_name == "arp-w":
                    tester = vav.HalfspaceVerificationTester(true_world, Qopt, opt_policy, debug = debug, precision=precision, teacher=teacher, tests=tests, halfspaces=halfspaces)
                
                elif verifier_name =="arp-bb":
                    tester = vav.ARPBlackBoxTester(true_world, Qopt, opt_policy, precision, debug=debug, teacher=teacher, tests=tests, halfspaces=halfspaces)

                elif verifier_name == "arp-pref":
                    tester = vav.TrajectoryRankingBasedTester(true_world, Qopt, opt_policy, precision, rollout_length, debug=debug, use_suboptimal_rankings=True)

                
                elif verifier_name == "scot":
                    tester = vav.SCOTVerificationTester(true_world, Qopt, opt_policy, precision, num_rollouts, rollout_length, debug=debug)
                
                else:
                    print("invalid verifier name")
                    sys.exit()
                size_verification_test = tester.get_size_verification_test()
                print("number of questions", size_verification_test)
                #checck optimal
                verified = tester.is_agent_value_aligned(opt_policy, Qopt, true_weights)

                #print(verified)
                if not verified:
                    print("testing true policy")
                
                    print("supposed to verify the optimal policy. This is not right!")
                    input()

                correct = 0
                for i in range(num_eval_policies):
                    
                    if debug:
                        print("\ntesting agent", i)
                        print("with reward weights:", eval_weights[i])
                        print("agent policy")
                        world.print_map(world.to_arrows(eval_policies[i]))
                        print("compared to ")
                        print("optimal policy")
                        true_world.print_map(true_world.to_arrows(opt_policy))
                        print("true reward weights:", true_weights)
                        print("mdp features")
                        utils.display_onehot_state_features(true_world)
                    verified = tester.is_agent_value_aligned(eval_policies[i], eval_Qvalues[i], eval_weights[i])
                    #print(verified)
                    if verified:
                        if debug:
                            print("not supposed to be true...")
                            input()
                    if not verified:
                        correct += 1
                #TODO: how do I keep track of accuracy??
                verifier_accuracy = correct / num_eval_policies
                print(verifier_name)
                print("Accuracy = ", 100.0*verifier_accuracy)
                #input()
                
                result_writers[vindx].write("{},{},{}\n".format(correct, num_eval_policies, size_verification_test))
        for writer in result_writers:
            writer.close()

    #teacher = machine_teaching.RankingTeacher(world, debug=False)
    #teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)