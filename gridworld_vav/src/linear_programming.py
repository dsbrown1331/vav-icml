#from cvxopt import matrix, solvers
import numpy as np

from scipy.optimize import linprog
import sys

# def toy_lp_scipy():
#     A = np.array([[-1., -1.]])
#     b = np.zeros(1)#np.array([ 1.0, -2.0, 0.0, 4.0 ])
#     c = np.array([ 1.0, 1.0 ])
#     sol = linprog(c, A_ub=A, b_ub = b, bounds=(-1,1) )
#     print(sol['fun'])
#     print(sol['status'])
#     #TODO: I wonder if simplex is better than interior point if things lie on a halfplane?
#     input()


# def toy_lp_numpy():
#     #solves min_x c^T x subject to Ax <= b
#     A = np.array([[-1., -1.]])
#     b = np.zeros(1)#np.array([ 1.0, -2.0, 0.0, 4.0 ])
#     c = np.array([ 1.0, 2.0 ])
#     A = matrix(A)  #note that these are columns!! Not rows!
#     b = matrix(b)
#     c = matrix(c)
#     print(c.size)
#     print(A.size)
#     print(b.size)
    
#     sol=solvers.lp(c,A,b)
#     print(sol['x'])
#     print(sol.keys())

# def example_lp_numpy():
#     #solves min_x c^T x subject to Ax <= b
#     A = np.transpose(np.array([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ]))
#     b = np.zeros(4)#np.array([ 1.0, -2.0, 0.0, 4.0 ])
#     c = np.array([ 2.0, 1.0 ])
#     A = matrix(A)  #note that these are columns!! Not rows!
#     b = matrix(b)
#     c = matrix(c)
#     print(c.size)
#     print(A.size)
#     print(b.size)
    
#     sol=solvers.lp(c,A,b)
#     print(sol['x'])
#     print(sol.keys())

# def example_lp():
#     #solves min_x c^T x subject to Ax <= b
#     A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])  #note that these are columns!! Not rows!
#     b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
#     c = matrix([ 2.0, 1.0 ])
#     print(c.size)
#     print(A.size)
#     print(b.size)
    
#     sol=solvers.lp(c,A,b)
#     print(sol['x'])
#     print(sol.keys())

#TODO: at some point should probably use something better than scipy, do we have a license for ibm's cplex solver?
def is_redundant_constraint(h, H, epsilon=0.0001):
    #we have a constraint c^w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to add c back into H 
    #Thus, we want to minimize c^T w subject to Hw >= 0
    #First we need to change this into the form min c^T x subject to Ax <= b
    #Our problem is equivalent to min c^T w subject to  -H w <= 0
    H = np.array(H) #just to make sure
    m,_ = H.shape
    #H = np.transpose(H)  #get it into correct format

    #c = matrix(h[non_zeros])
    #G = matrix(-H[:,non_zeros])
    b = np.zeros(m)
    sol = linprog(h, A_ub=-H, b_ub = b, bounds=(-1,1), method = 'revised simplex' )
    # print(sol)
    if sol['status'] != 0:
        print("trying interior point method")
        sol = linprog(h, A_ub=-H, b_ub = b, bounds=(-1,1) )
    
    if sol['status'] != 0: #not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue
        print("LP NOT SOLVABLE")
        print("IGNORING ERROR FOR NOW!!!!!!!!!!!!!!!!!!!")
        #sys.exit()
        return False #let's be safe and assume it's necessary...
    elif sol['fun'] < -epsilon: #if less than zero then constraint is needed to keep c^T w >=0
        return False
    else: #redundant since without constraint c^T w >=0
        #print("redundant")
        return True


# #buggy, doesn't seem to work due to weird rank constraint on constraints.
# #I wonder if I used simplex if this would work better?
# def is_redundant_constraint_cvxopt(h, H, epsilon=0.0001):
#     #we have a constraint c^w >= 0 we want to see if we can minimize c^T w and get it to go below 0
#     # if not then this constraint is satisfied by the constraints in H, if we can, then we need to add c back into H 
#     #Thus, we want to minimize c^T w subject to Hw >= 0
#     #First we need to change this into the form min c^T x subject to Ax <= b
#     #Our problem is equivalent to min c^T w subject to  -H w <= 0
#     m,_ = H.shape
#     #H = np.transpose(H)  #get it into correct format

    
#     c = matrix(h)
#     G = matrix(-H)
#     b = matrix(np.zeros(m))
#     solvers.options['show_progress'] = False #comment this to see solver logging messages and progress
#     sol = solvers.lp(c,G,b)
#     if sol['status'] != "optimal": #if infeasible then constraint is needed to keep c^T w >=0
#         return False
#     elif sol['primal objective'] < -epsilon: #if less than zero then constraint is needed to keep c^T w >=0
#         #Not sure if we need this case, but just to be safe...
#         return False
#     else: #redundant since without constraint c^T w >=0
        return True
    
def remove_redundant_constraints(halfspaces, epsilon = 0.0001):
    """Return a new array with all redundant halfspaces removed.

       Parameters
       -----------
       halfspaces : list of halfspace normal vectors such that np.dot(halfspaces[i], w) >= 0 for all i

       epsilon : numerical precision for determining if redundant via LP solution 

       Returns
       -----------
       list of non-redundant halfspaces 
    """
    #for each row in halfspaces, check if it is redundant
    #num_vars = len(halfspaces[0]) #size of weight vector
    non_redundant_halfspaces = []
    halfspaces_to_check = halfspaces
    for i,h in enumerate(halfspaces):
        #print("\nchecking", h)
        halfspaces_lp = [h for h in non_redundant_halfspaces] + [h for h in halfspaces_to_check[1:]]
        halfspaces_lp = np.array(halfspaces_lp)
     
        ##debuggin
        #print(halfspaces_lp)
        #print(np.linalg.matrix_rank(halfspaces_lp))
        #u, s, v = np.linalg.svd(halfspaces_lp)
        #print(s)
        #input()
        #print("halfspaces under consideration\n", halfspaces_lp)
        #print(np.linalg.matrix_rank(halfspaces_lp), "vs", num_vars)
        # if np.linalg.matrix_rank(halfspaces_lp) < num_vars: #check to make sure LP is well-posed
        #     #all remaining halfspaces are required #TODO: Check, but I think this is true since we first normalize and remove redundancies
        #     #non_redundant_halfspaces.append(h)
        #     non_redundant_halfspaces.extend(halfspaces_to_check)
        #     break

        if not is_redundant_constraint(h, halfspaces_lp, epsilon):
            #keep h
            #print("not redundant")
            non_redundant_halfspaces.append(h)
        else:
            pass
            ##print("redundant")
            
        halfspaces_to_check = halfspaces_to_check[1:]
    return non_redundant_halfspaces


if __name__=="__main__":
    
    #example halfspace constraints in 2-d: all halfspace constraints are assumed to be the normal vectors such that 
    #x^T halfspace[i] >= 0 for all i
    halfspaces = np.array([[-1.,  0.],
                [ 0.09950372, -0.99503719],
                [ 0.8752954,  -0.48358862],
                [ 0.9304349,  -0.36645723],
                [ 0.95434967, -0.29869164],
                [ 0.70710678, -0.70710678]])

    #check if a particular constraint is redundant
    H = halfspaces[1:]
    c = halfspaces[0]  
    print("is redundant", is_redundant_constraint(c,H))

    #remove all redundant halfspace constraints
    minimal_halfspaces = remove_redundant_constraints(halfspaces)
    print("minimal halfspaces")
    for h in minimal_halfspaces:
        print(h)
