import src.mdp as mdp
import numpy as np
import src.linear_programming as linear_programming
import src.utils as utils
import sys
from scipy.spatial import distance
from src.traj_pair import TrajPair
from src.linear_programming import is_redundant_constraint, remove_redundant_constraints





class StateActionRankingTeacher:
    """takes an mdp world and returns the optimal teaching solution to teach the MDP
        
    """
    def __init__(self, world, Q, opt_policy, epsilon = 0.0001, debug=False, remove_redundancy_lp = True):
        self.world = world
        self.precision = epsilon
        self.debug = debug
        self.remove_redundancy_lp = remove_redundancy_lp
        #print("self.debug", debug)
        #solve MDP
        if self.debug:
            print("rewards")
            world.print_rewards()
        #V = mdp.value_iteration(world, epsilon=epsilon)
        self.Q = Q#mdp.compute_q_values(world, V, eps=epsilon)
        if self.debug:
            V = mdp.value_iteration(world, epsilon=epsilon)
            print("values function")
            world.print_map(V)

        self.opt_policy = opt_policy#mdp.find_optimal_policy(world, Q=self.Q, epsilon=epsilon)
        if self.debug:
            print("optimal policy")
            world.print_map(world.to_arrows(self.opt_policy))
        self.sa_fcounts = mdp.calculate_sa_expected_feature_counts(self.opt_policy, world, epsilon=epsilon)
        #print("expected feature counts")
        #for s,a in self.sa_fcounts:
        #    print("state {} action {} fcounts:".format(s, world.to_arrow(a)))
        #    print(self.sa_fcounts[s,a])


    def compute_halfspace_normals(self, use_suboptimal_rankings, compare_optimal, epsilon_gap = 0.0):
        """ if use_suboptimal_rankings = False then it will only compute actions where preferred aciton is optimal
            if use_suboptimal_rankings = True, then it will find a BEC that could be much smaller than BEC(\pi^*) since it will consider rankings between all
            pairs of actions, even suboptimal ones. This will give the machine teaching set for teaching a ranking learner, I think...
            if compare_optimal = True, then we include optimal action comparisons and induce hyperplane constraints, currently with ARP we don't need these since either action is okay
            so we are okay with a reward function that picks one over the other since we have equal preferences.
            
            only keep halfspace constraints such that the better action is at least epsilon_gap better
        """
        
        #for each state compute \Phi(s,a) - \Phi(s,b) for all a and b such that Q*(s,a) >= Q*(s,b)
        half_spaces = []
        arrow = self.world.to_arrow
        for s in self.world.states:
            #print("Computing halfspaces for state", s)
            actions = self.world.actions(s)
                
            if use_suboptimal_rankings:
                #seach over all action pairs
                for i in range(len(actions)):
                    for j in range(i+1,len(actions)):
                        action_i = actions[i]
                        action_j = actions[j]
                        #print("comparing action", arrow(action_i), "to action", arrow(action_j))
                        #figure out which one has higher Q-value
                        if np.abs(self.Q[s,action_i] - self.Q[s,action_j]) < self.precision:
                            if compare_optimal: #check if we should add this
                                #print("action", arrow(action_i), "is equal to action", arrow(action_j))
                                normal_vector1 = self.sa_fcounts[s, action_i] - self.sa_fcounts[s, action_j]
                                normal_vector2 = self.sa_fcounts[s, action_j] - self.sa_fcounts[s, action_i]
                                #print("appending two normal vectors", normal_vector1)
                                #print("and", normal_vector1)
                                if np.linalg.norm(normal_vector1) > self.precision:
                                    half_spaces.append(normal_vector1)
                                    half_spaces.append(normal_vector2)
                        elif self.Q[s,action_i] > self.Q[s,action_j]:
                            #print("action", arrow(action_i), "is better")
                            normal_vector = self.sa_fcounts[s, action_i] - self.sa_fcounts[s, action_j]
                            #print("appending normal vector", normal_vector)
                            if np.linalg.norm(normal_vector) > self.precision:
                                half_spaces.append(normal_vector)
                        else:
                            #print("action", arrow(action_j), "is better")
                            normal_vector = self.sa_fcounts[s, action_j] - self.sa_fcounts[s, action_i]
                            #print("appending normal vector", normal_vector)
                            if np.linalg.norm(normal_vector) > self.precision:
                                half_spaces.append(normal_vector)
            else: #only consider optimal versus suboptimal halfspaces (what was done in AAAI'19 paper)
                #find optimal action(s) for s
                
                opt_actions = utils.argmax_list(actions, lambda a: self.Q[s,a], self.precision)
                for opt_a in opt_actions:
                    for action_b in actions:
                        if action_b in opt_actions:
                            if not compare_optimal:
                                #skip this (s,a) pair if we aren't comparing optimal actions
                                continue
                            
                        normal_vector = self.sa_fcounts[s, opt_a] - self.sa_fcounts[s, action_b]
                        #don't add if zero
                        if np.linalg.norm(normal_vector) > self.precision:
                            #don't add if not epsilon_gap better
                            if np.dot(normal_vector, self.world.weights) > epsilon_gap:
                                half_spaces.append(normal_vector)

        return half_spaces


    def preprocess_halfspace_normals(self, halfspace_normals):

        #I'm not going to normalize, I'm going to use cosine_dist to see if halfspaces are the same
        # #preprocess by normalizing all vectors
        # halfspace_normals = np.array(halfspace_normals) / np.linalg.norm(halfspace_normals, axis=1, keepdims=True)
        # if self.debug:
        #     print("normalized normals")
        #     for n in halfspace_normals:
        #         print(n)


        # #remove all zero vectors and duplicates
        # no_dups = []
        # for n in halfspace_normals:
        #     add_it = True
        #     if np.linalg.norm(n) < self.precision:
        #         print("error, zero vectors should already be removed")
        #         sys.exit()
        #         continue #skip since zero vector is unconstraining #Shouldn't ever get here
        #     else:
        #         for nd in no_dups:
        #             if np.linalg.norm(nd - n) < self.precision:
        #                 add_it = False
        #         if add_it:
        #             no_dups.append(n)
        # halfspace_normals = no_dups
        
        # print("unique normals")
        # for n in halfspace_normals:
        #     print(n)
        

        


        #preprocess by removing duplicates before running LP
        #use cosine_dist for similarity
        preprocessed_normals = []
        for n in halfspace_normals:
            already_in_list = False
            #search through preprocessed_normals for close match
            for pn in preprocessed_normals:
                if distance.cosine(n, pn) < self.precision:
                    already_in_list = True
                    break
            if not already_in_list:
                #add to list
                preprocessed_normals.append(n)

        if self.debug: 
            print("preprocessed normals before LP")
            for pn in preprocessed_normals:
                print(pn)

        #uncomment this and comment out below code to skip the linear programming part
        #min_constraints = preprocessed_normals
    
        #run linear programming to remove redundancies
        if len(preprocessed_normals) > 2 and self.remove_redundancy_lp:
            min_constraints = linear_programming.remove_redundant_constraints(preprocessed_normals)
        else:
            #don't need to run LP since only two halfspaces so neither will be redundant
            min_constraints = preprocessed_normals

        if self.debug:
            print("non-redundant constraints after running LP")
            for n in min_constraints:
                print(n)
        return min_constraints
        

    #TODO: dont' use this function but keep another running list of (s,a) pref (s,b) when I first compute halfspace normals then I can index into that 
    # and use the precomputed normals to test things...if too slow, I should look through my old code to see what optimizations I used. 

    def compute_all_tests(self, min_constraints, use_suboptimal_rankings):
        """ if use_suboptimal_rankings = False then it will only consider pairwise questions that compare optimal actions with other actions
            if use_suboptimal_rankings = True, then it will consider pairwise questions that consider rankings between all possible
                pairs of actions, even suboptimal ones. 
            
            returns a dictionary mapping constraints to lists of possible test questions to verify constraints
        """
        #Iterate through all preference pairs and keep track of which ones match which constraints
        #for each state compute \Phi(s,a) - \Phi(s,b) for all a and b such that Q*(s,a) >= Q*(s,b)
        test_questions=[[] for c in min_constraints] #list of list to hold questions that match each concept in min_constraints
        #questions are of form ((s,a), (s,b), [bool]) where True is optional, and relation is Q(s,b) = Q(s,a) if bool else Q(s,b) > Q(s,a) 
        print("finding relevant test questions")
        #print(test_questions)
        arrow = self.world.to_arrow #to make debugging actions human readable
        for s in self.world.states:
            if self.debug:
                print()
                print("Computing halfspaces for state", s)
            actions = self.world.actions(s)
                
            if use_suboptimal_rankings:
                #seach over all action pairs for possible test questions
                for i in range(len(actions)):
                    for j in range(i+1,len(actions)):
                        action_i = actions[i]
                        action_j = actions[j]
                        
                        if self.debug: print("comparing action", arrow(action_i), "to action", arrow(action_j))
                        #figure out which one has higher Q-value
                        if np.abs(self.Q[s,action_i] - self.Q[s,action_j]) < self.precision:
                            if self.debug: print("action", arrow(action_i), "is equal to action", arrow(action_j))
                            normal_vector1 = self.sa_fcounts[s, action_i] - self.sa_fcounts[s, action_j]
                            normal_vector2 = self.sa_fcounts[s, action_j] - self.sa_fcounts[s, action_i]
                            if self.debug:
                                print("trying normal vectors", normal_vector1)
                                print("and", normal_vector1)
                            #Remember: Test questions (i,j) means j preferred to i!
                            self.try_to_add_to_test(normal_vector1, ((s, action_j),(s,action_i)), test_questions, min_constraints)
                            self.try_to_add_to_test(normal_vector2, ((s,action_i),(s,action_j)), test_questions, min_constraints)
                        elif self.Q[s,action_i] > self.Q[s,action_j]:
                            if self.debug: print("action", arrow(action_i), "is better")
                            normal_vector = self.sa_fcounts[s, action_i] - self.sa_fcounts[s, action_j]
                            if self.debug: print("trying normal vector", normal_vector)
                            self.try_to_add_to_test(normal_vector, ((s,action_j),(s,action_i)), test_questions, min_constraints)
                        else:
                            if self.debug: print("action", arrow(action_j), "is better")
                            normal_vector = self.sa_fcounts[s, action_j] - self.sa_fcounts[s, action_i]
                            if self.debug: print("trying normal vector", normal_vector)
                            self.try_to_add_to_test(normal_vector, ((s,action_i), (s,action_j)), test_questions, min_constraints)
            else: #only consider optimal versus other halfspaces (what was done in AAAI'19 paper)
                #find optimal action(s) for s
                opt_actions = utils.argmax_list(actions, lambda a: self.Q[s,a], self.precision)
                for opt_a in opt_actions:
                    for action_b in actions:
                        if action_b not in opt_actions:
                            if self.debug: print("comparing opt action", arrow(opt_a), "to action", arrow(action_b))
                                
                            normal_vector = self.sa_fcounts[s, opt_a] - self.sa_fcounts[s, action_b]
                            if self.debug: print("trying", normal_vector)
                            self.try_to_add_to_test(normal_vector, ((s,action_b), (s,opt_a)), test_questions, min_constraints)
                        else:
                            #this is a potential equivalence query
                            normal_vector = self.sa_fcounts[s, opt_a] - self.sa_fcounts[s, action_b]
                            #we only try adding one direction, the other direction will be covered since we have a double for over all pairs
                            self.try_to_add_to_test(normal_vector, ((s,action_b), (s,opt_a), True), test_questions, min_constraints, equivalence=True)

        return test_questions  #list of lists of questions corresponding to list of constraints

    def try_to_add_to_test(self, normal_vector, test_question, test_question_lists, constraints, equivalence=False):
        #go through each constraint (key in dict) and see if it matches
        #need to normalize normal_vector! Check first to make sure not zero vector
        #if equivalence is True then try matching this normal with it's negation as well since w^T normal_vector = 0
        #hence w^T normal_vector >=0 and w^T -normal_vector <= 0
        
        #make sure we're not dealing with an all zeros normal.
        if np.sum(np.abs(normal_vector)) < self.precision:
            return #ignore this normal vector

        #check if it matches any of the constraints
        for i,c in enumerate(constraints):
            if self.debug: print("checking if matches constraint", c)
            if distance.cosine(c, normal_vector) < self.precision:
                if self.debug: print("Matches!. Adding question")
                #it's a good test question
                test_question_lists[i].append(test_question)


    def get_optimal_value_alignment_tests(self, use_suboptimal_rankings = False, compare_optimal=False, epsilon_gap = 0.0):
        
        #get raw halfspace normals for all action pairs at each state (only for ones that have greater than epsilon_gap in value diff)
        halfspace_normals = self.compute_halfspace_normals(use_suboptimal_rankings, compare_optimal, epsilon_gap)
        #np.random.shuffle(halfspace_normals)
        ##Debug
        if self.debug:
            print("raw halfspace constraints")
            for n in halfspace_normals:
                print(n)


        #preprocess them to remove any redundancies
        min_constraints = self.preprocess_halfspace_normals(halfspace_normals)
        
        ##Debug
        print(len(min_constraints), "non-redundant feature weight constraints after full preprocessing")
        for n in min_constraints:
            print(n)

        #don't need to do set cover since each pairwise preference only gives one halfspace, just need to match them up
        #TODO: what should we return? for now let's return all the solutions: a list of sets where if you pick one element from each set you get a
        #valid machine testing set of pairwise preference queries.

        #get optimal teaching test set for pairwise preference queries
        alignment_test_questions = self.compute_all_tests(min_constraints, use_suboptimal_rankings)
        #print(alignment_test_questions)
        
        ##Debug
        if self.debug:
            arrow = self.world.to_arrow #to make debugging actions human readable
            for i,c in enumerate(min_constraints):
                print("questions that cover concept", c)
                for question in alignment_test_questions[i]:
                    utils.print_question(question, self.world)

        return alignment_test_questions, min_constraints


class TrajectoryRankingTeacher(StateActionRankingTeacher):
    """takes an mdp world and returns the optimal teaching solution to teach the MDP
        solution is pairs of trajectories and preferences over them
        
    """
    
    
    def __init__(self, world, Q, opt_policy, precision, horizon, debug=False, use_suboptimal_rankings = False):
        super().__init__(world, Q, opt_policy, precision, debug)
        self.horizon = horizon #how long to rollout the demos
        self.use_suboptimal_rankings = use_suboptimal_rankings
        
    

    def get_optimal_value_alignment_tests(self):

        # #compute the AEC  #use this code for debugging and test the counter example

        # #get raw halfspace normals for all action pairs at each state
        # halfspace_normals = self.compute_halfspace_normals(use_suboptimal_rankings)
        # #np.random.shuffle(halfspace_normals)
        # ##Debug
        # if self.debug:
        #     print("raw halfspace constraints")
        #     for n in halfspace_normals:
        #         print(n)


        # #preprocess them to remove any redundancies
        # min_constraints = self.preprocess_halfspace_normals(halfspace_normals)
        
        # ##Debug
        # print(len(min_constraints), "non-redundant feature weight constraints after full preprocessing")
        # for n in min_constraints:
        #     print(n)

        #compute tests using trajectories (currently just taking all actions from each state and then following optimal policy)
        print("generating trajectory pairs")
        trajpairs = self.generate_trajectory_pairs(self.use_suboptimal_rankings)

        #compute the halfspace constraints for seeing if we can make all the expected feature count halfspaces redundant
        Hspace_trajs = []
        for t in trajpairs:
            if t.equivalence:
                #current code doesn't add if equivalent
                continue
                #need to add halfspace constraint both ways
                #Hspace_trajs.append(t.halfspace)
                #Hspace_trajs.append(-t.halfspace)
            else:
                Hspace_trajs.append(t.halfspace)
        Hspace_trajs = np.array(Hspace_trajs)
        #np.random.shuffle(Hspace_trajs)

        print("removing redundancies from trajectory halfspaces", len(Hspace_trajs))
        #okay, now remove the redundancies from Hspace_trajs
        H_minimal = np.array(self.preprocess_halfspace_normals(Hspace_trajs))
        print(H_minimal.shape[0], "remaining halfspaces afterwards")
        

        # #use this for debugging and testing the counter example
        # print("checking if we can make all expected halfspace constraints redundant")
        # #remove redundancies over original AEC based on expected feature counts
        # for h in min_constraints:
        #     #check if redundant
        #     print("checking constraint", h)
        #     if not is_redundant_constraint(h, H_minimal, self.precision):
        #         print("!!!!!!!!$$$$$%^^^^^^&***#############")
        #         print("error, this should be covered for exact alignment verification")
        #         print("this halfspace not covered", h)
        #         print("should be covered by")
        #         print(H_minimal)
        #         #input("continue?")

       

        # print("success: all are redundant!")
        
        #TODO: this step could probably be completed with computation of previous steps
        #now match trajectories until all H_minimal is covered.
        test_questions = []
        for h in H_minimal:
            #find first match in trajpairs
            for tp in trajpairs:
                if distance.cosine(h, tp.halfspace) < self.precision:
                    #match
                    test_questions.append(tp)
                    break
                if tp.equivalence:
                    #check negative direction too
                    if distance.cosine(h, -tp.halfspace) < self.precision:
                        test_questions.append(tp)
                        break



        return test_questions, H_minimal
    

    def generate_trajectory_pairs(self, use_suboptimal_rankings):
        #Iterate through all states and actions, currently using suboptimals Create TrajPair and return 
        trajectory_pairs = []
        arrow = self.world.to_arrow #to make debugging actions human readable
    
        for s in self.world.states:
            if self.debug:
                print()
                print("Computing trajectories for state", s)
            actions = self.world.actions(s)
            if use_suboptimal_rankings:    
                #seach over all action pairs for possible test questions
                for i in range(len(actions)):
                    for j in range(i+1,len(actions)):
                        action_i = actions[i]
                        action_j = actions[j]
                        
                        if self.debug: print("comparing traj with initial action", arrow(action_i), "to action", arrow(action_j))
                        
                        #create tuple
                        traj_i = utils.sa_optimal_rollout_from_Qvals(s, action_i, self.horizon, self.Q, self.world, self.precision)
                        traj_j = utils.sa_optimal_rollout_from_Qvals(s, action_j, self.horizon, self.Q, self.world, self.precision)
                        tpair = TrajPair(traj_i, traj_j, self.world, self.precision)
                        #check if non-zero since zero halfspace constraints are trivial
                        if np.linalg.norm(tpair.halfspace) > self.precision:
                            trajectory_pairs.append(tpair)
            else:
                opt_actions = utils.argmax_list(actions, lambda a: self.Q[s,a], self.precision)
                for opt_a in opt_actions:
                    for action_b in actions:
                        if not action_b in opt_actions:
                            traj_i = utils.sa_optimal_rollout_from_Qvals(s, opt_a, self.horizon, self.Q, self.world, self.precision)
                            traj_j = utils.sa_optimal_rollout_from_Qvals(s, action_b, self.horizon, self.Q, self.world, self.precision)
                            tpair = TrajPair(traj_i, traj_j, self.world, self.precision)
                            #check if non-zero since zero halfspace constraints are trivial
                            if np.linalg.norm(tpair.halfspace) > self.precision:
                                trajectory_pairs.append(tpair)
        #np.random.shuffle(trajectory_pairs)
        return trajectory_pairs
    


#TODO: Test this with stochastic dynamics?
class SCOT(StateActionRankingTeacher):
    def __init__(self, world, Q, opt_policy, precision, num_rollouts, rollout_length, compare_optimal, debug=False):
        super().__init__(world, Q, opt_policy, precision, debug)
        self.num_rollouts = num_rollouts
        self.rollout_length = rollout_length
        self.compare_optimal = compare_optimal  #this parameter allows us to either compute the AEC(true) or the ARP(false).

    def generate_candidate_trajectories(self):
        trajs = []
        for s in self.world.states:
            #check if initial state
            if s in self.world.initials:
                for k in range(self.num_rollouts):
                    traj = utils.optimal_rollout_from_Qvals(s, self.rollout_length, self.Q, self.world, self.precision)
                    #traj = mdp.generate_demonstration(s, self.opt_policy, self.world, self.rollout_length)
                    trajs.append(traj)
        return trajs


    #NOTE: this doesn't remove redundancies but does remove duplicates
    def get_all_constraints_traj(self, traj):
        constraints = []
        for s,a in traj:
            if s not in self.world.terminals: #don't need to worry about terminals since all actions self loop with zero reward
                for b in self.world.actions(s):
                    normal_vector = self.sa_fcounts[s, a] - self.sa_fcounts[s, b]
                    #don't add if zero
                    if np.linalg.norm(normal_vector) > self.precision:
                        constraints.append(normal_vector)

        #preprocess by removing duplicates before running LP
        #use cosine_dist for similarity
        preprocessed_normals = []
        for n in constraints:
            already_in_list = False
            #search through preprocessed_normals for close match
            for pn in preprocessed_normals:
                if distance.cosine(n, pn) < self.precision:
                    already_in_list = True
                    break
            if not already_in_list:
                #add to list
                preprocessed_normals.append(n)

        if self.debug: 
            print("preprocessed normals before LP")
            for pn in preprocessed_normals:
                print(pn)

        return preprocessed_normals    

    def count_new_covers(self, constraints_new, constraint_set, covered):
        #go through each element of constraints_new and see if it matches an uncovered element of constraint_set
        count = 0
        for c_new in constraints_new:
            for i,c in enumerate(constraint_set):
                #check if equal via cosine dist
                if distance.cosine(c_new, c) < self.precision:
                    #check if not covered yet
                    if not covered[i]:
                        count += 1
        return count

    def update_covered_constraints(self, constraints_added, constraint_set, covered):
        for c_new in constraints_added:
            for i,c in enumerate(constraint_set):
                #check if equal via cosine dist
                if distance.cosine(c_new, c) < self.precision:
                    #check if not covered yet
                    if not covered[i]:
                        covered[i] = True
        return covered

    def get_machine_teaching_demos(self):
        use_suboptimal_rankings = False
        #get raw halfspace normals for all action pairs at each state
        halfspace_normals = self.compute_halfspace_normals(use_suboptimal_rankings, self.compare_optimal)
        #np.random.shuffle(halfspace_normals)
        ##Debug
        if self.debug:
            print("raw halfspace constraints")
            for n in halfspace_normals:
                print(n)


        #preprocess them to remove any redundancies
        constraint_set = self.preprocess_halfspace_normals(halfspace_normals)
        
        ##Debug
        print(len(constraint_set), "non-redundant feature weight constraints after full preprocessing")
        for n in constraint_set:
            print(n)



        #generate k trajectories of length H from each start state
        candidate_trajs = self.generate_candidate_trajectories()

        #create boolean bookkeeping to see what has been covered in the set
        covered = [False for _ in constraint_set]
        
        #for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
        total_covered = 0
        opt_demos = []
        while total_covered < len(constraint_set):
            constraints_to_add = None
            best_traj = None
            max_count = 0
            for traj in candidate_trajs:
                #TODO: optimize by precomputing and saving this
                constraints_new = self.get_all_constraints_traj(traj)
 
                count = self.count_new_covers(constraints_new, constraint_set, covered)
                if self.debug: print("covered", count)
                if count > max_count:
                    max_count = count
                    constraints_to_add = constraints_new
                    best_traj = traj

            #update covered flags and add best_traj to demo`
            opt_demos.append(best_traj)
            covered = self.update_covered_constraints(constraints_to_add, constraint_set, covered)
            total_covered += max_count
            #TODO: optimize by removing trajs if we decide to add to opt_demos
    
        return opt_demos



class MdpFamilyTeacher(SCOT):
    '''
    Takes as input a family of MDPs (list of mdps)
    calculates the AEC for each MDP and then runs LP to remove redundancies
    then runs set cover using entire MDPs to cover the set of halfspaces
    returns the approximately minimal set of MDPs to test/teach on

    '''
    
    def __init__(self, mdp_family, precision, use_suboptimal_rankings, compare_optimal, debug=False):
        self.mdp_family = mdp_family
        self.precision = precision
        self.debug = debug
        self.mdp_halfspaces = []
        self.compare_optimal
        all_halfspaces = []
        for i,mdp_world in enumerate(mdp_family):
            #print("\n",i)
            if self.debug: print(mdp_world.features)
            #get all halfspace constraints
            mteacher = StateActionRankingTeacher(mdp_world, epsilon = precision, debug=debug)
            halfspace_normals = mteacher.compute_halfspace_normals(use_suboptimal_rankings, compare_optimal)
            #accumulate halfspaces
            halfspaces = mteacher.preprocess_halfspace_normals(halfspace_normals)
            self.mdp_halfspaces.append(halfspaces)
            if self.debug: print(halfspaces)
            all_halfspaces.extend(halfspaces)
        all_halfspaces = np.array(all_halfspaces)
        print("all before processing")
        print(all_halfspaces)
        #remove redundancies except for lp
         #preprocess by removing duplicates before running LP
        #use cosine_dist for similarity
        preprocessed_normals = []
        for n in all_halfspaces:
            already_in_list = False
            #search through preprocessed_normals for close match
            for pn in preprocessed_normals:
                if distance.cosine(n, pn) < self.precision:
                    already_in_list = True
                    break
            if not already_in_list:
                #add to list
                preprocessed_normals.append(n)
        self.all_halfspaces = np.array(preprocessed_normals)
        #run linear programming to remove redundancies
        if len(preprocessed_normals) > 2:
            min_constraints = linear_programming.remove_redundant_constraints(preprocessed_normals)
        else:
            #don't need to run LP since only two halfspaces so neither will be redundant
            min_constraints = preprocessed_normals

        #family_halfspaces = mteacher.preprocess_halfspace_normals(preprocessed_normals)
        self.family_halfspaces = np.array(min_constraints)
        print(self.family_halfspaces)
        #input()

    def get_halfspaces_for_plotting(self):
        minimal_set = []
        for i,h in enumerate(self.all_halfspaces):
            for hj in self.family_halfspaces:
                if distance.cosine(h, hj) < self.precision:
                    minimal_set.append(i)
        return self.all_halfspaces, minimal_set


    def get_machine_teaching_mdps(self):
        
        constraint_set = self.family_halfspaces
        candidate_mdps = self.mdp_family
        candidate_halfspaces = self.mdp_halfspaces
        #create boolean bookkeeping to see what has been covered in the set
        covered = [False for _ in constraint_set]
        
        #for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
        total_covered = 0
        opt_mdps = []
        while total_covered < len(constraint_set):
            if self.debug: print("set cover iteration")
            constraints_to_add = None
            best_mdp = None
            max_count = 0
            for i, mdp_env in enumerate(candidate_mdps):
                # if self.debug:
                #     print("-"*20) 
                #     print("MDP", i)

                #     V = mdp.value_iteration(mdp_env, epsilon=self.precision)
                #     Qopt = mdp.compute_q_values(mdp_env, V=V, eps=self.precision)
                #     opt_policy = mdp.find_optimal_policy(mdp_env, Q = Qopt, epsilon=self.precision)
                #     print("rewards")
                #     mdp_env.print_rewards()
                #     print("value function")

                #     mdp_env.print_map(V)
                #     print("mdp features")
                #     utils.display_onehot_state_features(mdp_env)

                #     print("optimal policy")
                #     mdp_env.print_map(mdp_env.to_arrows(opt_policy))

                #     print("halfspace")
                #     print(candidate_halfspaces[i])
                #get the halfspaces induced by an optimal policy in this MDP
                constraints_new = candidate_halfspaces[i]
 
                count = self.count_new_covers(constraints_new, constraint_set, covered)
                #if self.debug: print("covered", count)
                if count > max_count:
                    max_count = count
                    constraints_to_add = constraints_new
                    best_mdp = mdp_env
                    if self.debug:
                        print()
                        print("best mdp so far")
                        print("-"*20) 
                        print("MDP", i)

                        V = mdp.value_iteration(mdp_env, epsilon=self.precision)
                        Qopt = mdp.compute_q_values(mdp_env, V=V, eps=self.precision)
                        opt_policy = mdp.find_optimal_policy(mdp_env, Q = Qopt, epsilon=self.precision)
                        print("rewards")
                        mdp_env.print_rewards()
                        print("value function")

                        mdp_env.print_map(V)
                        print("mdp features")
                        utils.display_onehot_state_features(mdp_env)

                        print("optimal policy")
                        mdp_env.print_map(mdp_env.to_arrows(opt_policy))

                        print("halfspace")
                        print(constraints_to_add)

                        print("covered", count)

            #update covered flags and add best_traj to demo`
            opt_mdps.append(best_mdp)
            covered = self.update_covered_constraints(constraints_to_add, constraint_set, covered)
            total_covered += max_count
            #TODO: optimize by removing trajs if we decide to add to opt_demos
    
        return opt_mdps