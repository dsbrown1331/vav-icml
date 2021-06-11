import numpy as np
import src.utils as utils
import copy
import random

#the following code is adapted from  erensezener/aima-based-irl 

class LinearFeatureGridWorld:
    """A Markov Decision Process, defined by an initial state distribution, 
    transition model, reward function, gamma, and action list"""

    def __init__(self, features, weights, initials, terminals, gamma=.95):
        self.features = features
        self.weights = copy.copy(weights)
        self.initials=initials #assumes uniform distribution
        self.actlist=[(0,-1),(-1,0), (0,1), (1,0)  ] #up, down, left, right
        self.terminals=terminals
        self.gamma=gamma
        self.rows, self.cols = len(features), len(features[0])
        self.states = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if features[r][c] is not None: #if features are None then there is an obstacle
                    self.states.add((r,c))

        #human readable action mappings
        self.chars = {(1, 0): 'v', (0, 1): '>', (-1, 0): '^', (0, -1): '<', None: '.'}
        
    def get_num_features(self):
        return len(self.weights)

    def get_state_features(self, state):
        r,c = state
        return self.features[r][c]

    def R(self, state):
        "Return a numeric reward for this state."
        r,c = state
        if self.features[r][c] is None: #wall or inaccessible state due to barrier
            return None
        else:
            return np.dot(self.features[r][c], self.weights)

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(1.0, self.go(state, action))]

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a 
        fixed list of actions, except for terminal states. Override this 
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = self.vector_add(state, direction)
        if state1 in self.states: #move if a valid next state (i.e., not off the grid or into obstacle)
            return state1
        else: #self transition
            return state

    def vector_add(self, a, b):
        """Component-wise addition of two vectors.
        >>> vector_add((0, 1), (8, 9))
        (8, 10)
        """
        return a[0] + b[0], a[1] + b[1]

    def to_arrow(self, action):
        return self.chars[action]


    def to_arrows(self, policy):
        policy_arrows = {}
        for (s,a_list) in policy.items():
            #concatenate optimal actions
            opt_actions = ""
            for a in a_list:
                opt_actions+=self.chars[a]
            policy_arrows[s] = opt_actions
        return policy_arrows

    def print_policy(self, policy):
        arrow_map = self.to_arrows(policy)


    def to_grid(self, mapping):
        """Convert a mapping from (r, c) to val into a [[..., val, ...]] grid."""
        return list([[mapping.get((r, c), None)
                               for c in range(self.cols)]
                              for r in range(self.rows)])

    def print_2darray(self, array_2d):
        """take a 2-d array of values and print nicely"""
        for r in (range(self.rows)):
            for c in (range(self.cols)):
                if array_2d[r][c] is None:
                    print("{}".format(array_2d[r][c], 3), end="\t")
                elif type(array_2d[r][c]) is int:
                    print("{}".format(array_2d[r][c], 3), end="\t")
                elif type(array_2d[r][c]) is str:
                    print("{}".format(array_2d[r][c], 3), end="\t")
                else:
                    print("{:0.3f}".format(array_2d[r][c], 3), end="\t")
                
                    
            print()

    
         

    def print_map(self, mapping):
        array2d = self.to_grid(mapping)
        self.print_2darray(array2d)


    def print_rewards(self):
        for r in (range(self.rows)):
            for c in (range(self.cols)):
                reward = self.R((r,c))
                #print((r,c), end="\t")
                if reward is None: #wall obstacle
                    print("None", end="\t")
                else:
                    print("{:0.2f}".format(reward), end="\t")
            print("",end="\n")

    def get_grid_size(self):
        return len(self.grid), len(self.grid[0])


#______________________________________________________________________________


def calculate_trajectory_feature_counts(traj, mdp_world):
    gamma = mdp_world.gamma
    num_features = mdp_world.get_num_features()
    traj_features = np.zeros(num_features)
    for t, s_a in enumerate(traj):
        s,a = s_a
        traj_features += gamma ** t  * np.array(mdp_world.get_state_features(s))
    return traj_features


def calculate_expected_feature_counts(pi, mdp, epsilon=0.0001):
    """run policy evaluation but keep track of k-dimensional feature vectors rather than scalar values"""
    T, gamma = mdp.T, mdp.gamma
    num_features = mdp.get_num_features()
    fcounts = dict([(s,np.zeros(num_features)) for s in mdp.states])
    #print("num features", num_features)
    while True:
        #print "k", i
        delta = 0
        for s in mdp.states:
            #print "s", s
            #accumulate expected features of successor states
            sum_next_state_features = np.zeros(num_features)
            for opt_action in pi[s]:
                for p,s1 in T(s, opt_action):
                    sum_next_state_features += p * fcounts[s1]
                    #print(sum_next_state_features)
            sum_next_state_features /= len(pi[s])  #normalize since we assume equal probability of all optimal actions
            #updated estimate
            updated_fcounts = mdp.get_state_features(s) + gamma * sum_next_state_features
            delta = max(delta, max(abs(fcounts[s] - updated_fcounts))) #check max change along feature vector
            fcounts[s] = updated_fcounts
        if delta < epsilon * (1 - gamma) / gamma:
            return fcounts

def calculate_sa_expected_feature_counts(pi, mdp, epsilon=0.0001):
    """return dictionary of feature counts associated with each (s,a) pair"""
    sa_fcounts = dict()
    #compute feature expectations per state
    fcounts = calculate_expected_feature_counts(pi, mdp, epsilon)
    #(s,a) feature expectations are \phi(s) + \gamma * \sum_s' T(s,a,s') * fcounts[s']
    for s in mdp.states:
        for a in mdp.actions(s):
            sa_fcounts[s,a] = mdp.get_state_features(s)
            for p,s1 in mdp.T(s, a):
                sa_fcounts[s,a] += mdp.gamma * p * fcounts[s1]
    return sa_fcounts




def value_iteration_inplace(mdp, epsilon=0.0001):
    "Solving an MDP by value iteration."
    V = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    
    while True:
        delta = 0
        for s in mdp.states:
            updated_value = R(s) + gamma * max([sum([p * V[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(V[s] - updated_value))
            V[s] = updated_value
            
        #print V1
        if delta < epsilon * (1 - gamma) / gamma:
            return V


def value_iteration(mdp, epsilon=0.0001):
    "Solving an MDP by value iteration."
    V1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        V = V1.copy()
        delta = 0
        for s in mdp.states:
            V1[s] = R(s) + gamma * max([sum([p * V[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(V1[s] - V[s]))
        #print V1
        if delta < epsilon * (1 - gamma) / gamma:
            return V1

#evaluate a policy under an MDP, return values per state
def policy_evaluation(pi, mdp, epsilon=0.0001):
    "Solving an MDP by value iteration."
    V1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        V = V1.copy()
        delta = 0
        for s in mdp.states:
            V1[s] = R(s) + gamma * sum([p * V[s1] for (p, s1) in T(s, pi[s][0])])
                                        
            delta = max(delta, abs(V1[s] - V[s]))
        #print V1
        if delta < epsilon * (1 - gamma) / gamma:
            return V1




def compute_q_values(mdp, V=None, eps = 0.0001):
    if not V:
        #first we need to compute the value function
        V = value_iteration(mdp, epsilon=eps)
    Q = {}
    for s in mdp.states:
        for a in mdp.actions(s):
            Qtemp = mdp.R(s)
            for (p, sp) in mdp.T(s, a):
                Qtemp += mdp.gamma * p * V[sp]
            Q[s, a] = Qtemp
    return Q



def find_optimal_policy(mdp, V=None, Q=None, epsilon=0.0001):
    """Given an MDP and an optional value function V or optional Q-value function, determine the best policy,
    as a mapping from state to action."""
    #check if we need to compute Q-values
    if not Q:
        if not V:
            Q = compute_q_values(mdp, eps = epsilon)
        else:
            Q = compute_q_values(mdp, V, eps=epsilon)

    pi = {}
    for s in mdp.states:
        #find all optimal actions

        pi[s] = utils.argmax_list(mdp.actions(s), lambda a: Q[s,a], epsilon)
    return pi


def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


#______________________________________________________________________________


#TODO what is a good value for k?
def policy_evaluation_old(pi, U, mdp, k=100):
    """Return an updated utility mapping U from each state in the MDP to its 
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        #print "k", i
        for s in mdp.states:
            #print "s", s
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
            #print "U[s]", U[s]
    return U

#how to generate a demo from a start location
###TODO  this is for deterministic settings!!
###TODO add a terminationg criterion like value and policy iteration!
def generate_demonstration(start, policy, mdp_world, horizon):
    """given a start location return the demonstration following policy
    return a state action pair array
    
    trajectory is of length horizon
    
    """
    
    demonstration = []
    curr_state = start
    #print('start',curr_state)
    steps = 0
    while curr_state not in mdp_world.terminals and steps < horizon:
        #print('actions',policy[curr_state])
        a = random.sample(policy[curr_state],1)[0]
        #print(a)
        demonstration.append((curr_state, a))
        curr_state = mdp_world.go(curr_state, a)
        steps += 1
        #print('next state', curr_state)
    if curr_state in mdp_world.terminals:
        #append the terminal state
        demonstration.append((curr_state, None))
    return demonstration