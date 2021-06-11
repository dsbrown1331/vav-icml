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
import src.grid_worlds as gw
import src.value_alignment_verification as vav
import src.alignment_heuristics as ah
import data_analysis.plot_grid as mdp_plot


seed = 1222
np.random.seed(seed)
import random
random.seed(seed)

world = gw.create_safety_lava_world_nowalls()

print("rewards")
world.print_rewards()
V = mdp.value_iteration(world)
Q = mdp.compute_q_values(world, V)
print("values")
world.print_map(V)

opt_policy = mdp.find_optimal_policy(world, Q=Q)
print("optimal policy")
world.print_map(world.to_arrows(opt_policy))
print(opt_policy)

lava_colors = ['black','tab:green','white','tab:red','tab:blue','tab:gray','tab:green','tab:purple', 'tab:orange',  'tab:cyan']

mdp_plot.plot_optimal_policy_vav(opt_policy, world.features, walls=True, show=False, arrow_color='k',
    feature_colors=lava_colors,filename=os.path.join(project_path,'figs/lava/optimal.png'))

#what does the test look like? can we visualize it?

#let's look at the heuristics:

debug = False
precision = 0.00001
rollout_length = 30
num_rollouts = 10
critical_value_thresh = 10.0

#verifier_list =["arp-bb","scot", "arp-w","state-value-critical-0.2"]
###arp-bb
tester = vav.ARPBlackBoxTester(world, precision, debug)
size_verification_test = tester.get_size_verification_test()
print("number of questions", size_verification_test)

arp_halfspaces = np.array(tester.halfspaces)

print("all questions")
for questions in tester.tests:
    print(questions)

print("arp-bb test questions")
for question in tester.test:
    utils.print_question(question, world)





print("tests")
initials_test = []
for test in tester.tests:
    found = False
    #find something that starts in initial state if possible
    for question in test:
        (s,worse), (s,better) = question
        if s in world.initials:
            initials_test.append(question)
            found = True
            break
    if not found:
        print("Error")
        import sys
        sys.exit()


print(initials_test)
print("arp-bb inital state test questions")
for question in initials_test:
    utils.print_question(question, world)

arp_bb_question_list = []
for question in initials_test:
    (s,worse), (s,better) = question
    arp_bb_question_list.append((s,better))

mdp_plot.plot_test_questions(arp_bb_question_list, world.features, walls=True, show=False, arrow_color='k',
    feature_colors=lava_colors,filename=os.path.join(project_path,'figs/lava/arb-bb.png'))

###arp-pref
tester = vav.TrajectoryRankingBasedTester(world, precision, rollout_length, debug, use_suboptimal_rankings=True)
size_verification_test = tester.get_size_verification_test()
print("number of questions", size_verification_test)

print("arp-pref test questions")
for question in tester.test:
    print(question)
    
pref_halfspaces = np.array(tester.halfspaces)

for i, question in enumerate(tester.test):
    mdp_plot.plot_preference_query(question.traj_better, question.traj_worse, world.features, walls=True, show=False, 
        good_arrow_color='k', bad_arrow_color='orange', feature_colors=lava_colors,
        filename=os.path.join(project_path,'figs/lava/pref_{}.png'.format(i)))



###SCOT
tester = vav.SCOTVerificationTester(world, precision, num_rollouts, rollout_length, debug)
size_verification_test = tester.get_size_verification_test()
print("number of questions", size_verification_test)

print("SCOT test questions")
for s,a in tester.query_sa_pairs:
    print("Testing teaching state: ({}, {})".format(s, world.to_arrow(a)))      


scot_question_list = []
for sa_pair in tester.query_sa_pairs:
    scot_question_list.append(sa_pair)

mdp_plot.plot_test_questions(scot_question_list, world.features, walls=True, show=False, arrow_color='k',
    feature_colors=lava_colors,filename=os.path.join(project_path,'figs/lava/scot.png'))


###Critical States
tester = ah.CriticalStateActionValueVerifier(world, critical_value_thresh, precision, debug)
size_verification_test = tester.get_size_verification_test()
print("number of questions", size_verification_test)

print("Critical State test questions")
for s,a_list in tester.critical_state_actions:
    print("Testing critical state: ({}), actions {}".format(s, [world.to_arrow(a) for a in a_list]))
        

cs_question_list = []
for sa_pair in tester.critical_state_actions:
    cs_question_list.append(sa_pair)

mdp_plot.plot_test_questions(cs_question_list, world.features, walls=True, show=False, arrow_color='k',
    feature_colors=lava_colors,filename=os.path.join(project_path,'figs/lava/cs-10.png'))


#test whether the preference trajectories cover a smaller region than the ARP

def mc_volume(H, num_samples, rand_seed=None):
    #estimate volume of intersection of halfspaces via MC sampling

    if rand_seed:
        np.random.seed(rand_seed)
    num_constraints, num_vars = H.shape
    
    r = 1 - 2*np.random.rand(num_vars, num_samples)
    #print(r)
    bool_check = np.dot(H, r)>0
    check_cols = np.sum(bool_check.astype(int), axis=0)
    intersection = np.sum(check_cols == num_constraints)
    good_samples = r[:,check_cols == num_constraints]

    #return volume estimate and samples inside intersection
    return intersection / num_samples, good_samples

#test to make sure we get tighter constraints with arp-prefs
print(mc_volume(arp_halfspaces, 100000)[0])
print(mc_volume(pref_halfspaces, 100000)[0])

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

