import src.mdp as mdp
import numpy as np
import src.linear_programming as linear_programming
import src.utils as utils
import sys
from scipy.spatial import distance
from src.traj_pair import TrajPair
from src.linear_programming import is_redundant_constraint, remove_redundant_constraints

from sf_gym import rollout_halfspaces





class StateActionRankingTeacher:
    """takes a gym env and a tester (teacher) pretrained policy, runs rollouts to estimate successor features, and 
    removes duplicates of successor feature differences and returns the optimal teaching solution to teach the MDP
        
        epsilon here is just a numerical precision, but since we don't have a true reward weight vector we can also use it as 
        TODO: a value gap, maybe we'll add a true value gap later...
    """
    def __init__(self, env_id, policy, num_samples=100, epsilon = 0.0001, debug=False, remove_redundancy_lp = True):
        self.world = env_id

        #rollout pretrained teacher policy to generate halfspace constraints
        self.sa_halfspaces = rollout_halfspaces(env_id, policy, num_samples, precision=epsilon)
        self.halfspaces = [self.sa_halfspaces[i] for i in self.sa_halfspaces]
        self.precision = epsilon
        self.debug = debug
        self.remove_redundancy_lp = remove_redundancy_lp
        
       

    def preprocess_halfspace_normals(self, halfspace_normals):

        #I'm not going to normalize, I'm going to use cosine_dist to see if halfspaces are the same
        
        #preprocess by removing duplicates before running LP
        #use cosine_dist for similarity
        preprocessed_normals = []
        for n in halfspace_normals:
            already_in_list = False
            #search through preprocessed_normals for close match
            for pn in preprocessed_normals:
                print("cosine dist", n, pn, distance.cosine(n, pn))
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
        halfspace_normals = self.halfspaces
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



if __name__ == "__main__":
    env_id='CartPole-v1'
    policy='dqn'
    num_samples=10
    debug = True


    teacher = StateActionRankingTeacher(env_id, policy, num_samples=num_samples, debug=debug)
    teacher.get_optimal_value_alignment_tests()
