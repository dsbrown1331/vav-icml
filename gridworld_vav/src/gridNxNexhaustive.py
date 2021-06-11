"""
#okay so I want to generate all possible 3x3 grid worlds.

#Just generate an integer matrix and then perform the following transformations for all symmetries:
    unchanged:
    T : transpose
    H: reflect horizontally
    V: reflect vertically
    VH: vertical then horizontal
    TV: transpose then vertical reflection
    TH: tranpose then horizontal reflection
    TVH: tranpose then vertical then horizontal

    I need to generate all possible choices for all possible features and then remove symmetries to make solving MDPs faster

"""

import numpy as np
#from ordered_set import OrderedSet as set

def not_in_mdp_list_old(x,xt, mdp_list):
    for m,term in mdp_list:
        if np.all(m == x) and xt == term:
            return False
        if np.all(m == x.transpose())  and xt == term:
            return False
        if np.all(m == np.flipud(x))  and xt == term:
            return False
        if np.all(m == np.fliplr(x))  and xt == term:
            return False
        if np.all(m == np.flipud(np.fliplr(x)))  and xt == term:
            return False
        if np.all(m == np.flipud(x.transpose()))  and xt == term:
            return False
        if np.all(m == np.fliplr(x.transpose()))  and xt == term:
            return False
        if np.all(m == np.flipud(np.fliplr(x.transpose())))  and xt == term:
            return False
    return True

def hash_mdp(mdp_grid, terminal):
    #create hashable version
    to_hash = list(mdp_grid.flatten())
    term_hash = list(terminal.flatten())
    to_hash.extend(term_hash)
    to_hash = tuple(to_hash)
    #print(type(to_hash))
    return to_hash

def not_in_mdp_list(x, xt, mdp_set):
    hashed_symmetries = []
    hashed_symmetries.append(hash_mdp(x, xt))
    hashed_symmetries.append(hash_mdp(x.transpose(), xt.transpose()))
    hashed_symmetries.append(hash_mdp(np.flipud(x), np.flipud(xt)))
    hashed_symmetries.append(hash_mdp(np.fliplr(x), np.fliplr(xt)))
    hashed_symmetries.append(hash_mdp(np.flipud(np.fliplr(x)), np.flipud(np.fliplr(xt))))
    hashed_symmetries.append(hash_mdp(np.flipud(x.transpose()), np.flipud(xt.transpose()) ))
    hashed_symmetries.append(hash_mdp(np.fliplr(x.transpose()), np.fliplr(xt.transpose())))
    hashed_symmetries.append(hash_mdp(np.flipud(np.fliplr(x.transpose())), np.flipud(np.fliplr(xt.transpose()))))

    for h in hashed_symmetries:
        if h in mdp_set:
            return False

    return True

def print_hashed_mdp_term(h, grid_length):
    mdp_grid = h[: grid_length * grid_length]
    term_grid = h[grid_length * grid_length :]
    print(np.reshape(mdp_grid, (grid_length, grid_length)))
    print(np.reshape(term_grid, (grid_length, grid_length)))

def unhash_mdp(h, grid_length):
    mdp_grid = h[: grid_length * grid_length]
    term_grid = h[grid_length * grid_length :]
    mdp_grid = np.reshape(mdp_grid, (grid_length, grid_length))
    term_grid = np.reshape(term_grid, (grid_length, grid_length))
    
    return mdp_grid, term_grid



def get_all_unique_mdps(num_features, grid_length, terminal=True, max_mdps=False):
    #How to generate all possible MDPs? Breadth-first graph search.
    #include a single terminal state

    MDP_list = set() #visited
    start_mdp = np.zeros((grid_length, grid_length), dtype = np.uint8)
    start_mdp
    #print(start_mdp)
    frontier_nodes = [] # queue

    # add start all zero-th feature with all possible terminals
    for r in range(grid_length):
        for c in range(grid_length):
            term = (r,c)
            term_matrix = np.zeros((grid_length, grid_length), dtype=np.uint8)
            if terminal:
                term_matrix[term] = 1
            to_hash = hash_mdp(start_mdp, term_matrix)
            #print(to_hash)
            #print(type(to_hash))
            if not_in_mdp_list(start_mdp, term_matrix, MDP_list):
                MDP_list.add(to_hash)
                frontier_nodes.append((start_mdp, term_matrix))
         




    done = False
    while frontier_nodes and not done:
        s, term = frontier_nodes.pop(0) 
        #print ("expanding from", s) 

        #compute neighbors 
        #first add all the one step nodes from current nodes
        neighbors = []
        for r in range(grid_length):
            for c in range(grid_length):
                for f in range(num_features):
                    m = s.copy()
                    m[r][c] = f
                    neighbors.append((m, term))

        for neighbour, term in neighbors:
            if not_in_mdp_list(neighbour, term, MDP_list):
                MDP_list.add(hash_mdp(neighbour, term))
                frontier_nodes.append((neighbour, term))
                if max_mdps is not False and len(MDP_list) == max_mdps:
                    done = True
                    break

        if len(MDP_list) % 1000 == 0:
            print(len(MDP_list))
    print("all unique mdps")
    unique_mdps = []
    for m in MDP_list:
        
        #print(m)    
        #print_hashed_mdp_term(m, grid_length)
        mdp_grid, term_grid = unhash_mdp(m, grid_length)
        unique_mdps.append((mdp_grid, term_grid))
        #print("-"*10)
        #print("mdp\n", mdp_grid)
        #print("term\n", term_matrix)
    print("number of unique MDPS = ", len(MDP_list))
    
    return unique_mdps

def categorical_to_one_hot_features(categorical_features, num_features):
    #assumes a 2d grid of numbers for the features. 
    #returns a 2d grid of tuple vectors (one-hot features)
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]
    one_hots = []
    for row in categorical_features:
        f_row = []
        for el in row:
            f_row.append(features[el])
        one_hots.append(f_row)
    return one_hots


def get_terminals_from_grid(terminal_grid):
    #assumes a one hot grid denoting which states are terminals (1's)
    #returns tuples of (row,col) for the terminals
    #print(terminal_grid)
    num_rows, num_cols = terminal_grid.shape
    terminals = []
    for r in range(num_rows):
        for c in range(num_cols):
            if terminal_grid[r][c] == 1:
                terminals.append((r,c))

    return terminals



if __name__=="__main__":
    num_features = 2
    grid_length = 4
    unique_mdps = get_all_unique_mdps(num_features, grid_length, terminal=False)