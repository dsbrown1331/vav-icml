#class for keeping track of trajectory pairs and relevant information for preferences over them and halfspace constraints induced by them
import numpy as np
import src.utils as utils
import src.mdp as mdp
class TrajPair:
    #container class for holding trajs, halfspace, and boolean indicator of whether it's really a hyperplane
    def __init__(self, traj_A, traj_B, mdp_world, precision):
        self.mdp_world = mdp_world
        #compute feature counts
        traj_A_fcounts = mdp.calculate_trajectory_feature_counts(traj_A, mdp_world)
        traj_B_fcounts = mdp.calculate_trajectory_feature_counts(traj_B, mdp_world)
        #calculate the returns via dot product to see which is preferred
        return_diff = np.dot(traj_A_fcounts - traj_B_fcounts, mdp_world.weights)
        # print("return diff", return_diff)
        self.equivalence = False
        if abs(return_diff) < precision:
            #they are equal preference so doesn't matter which is better, just do it alphabetical :)
            self.traj_worse = traj_A
            self.traj_better = traj_B
            self.traj_worse_return = np.dot(traj_A_fcounts, mdp_world.weights)
            self.traj_better_return = np.dot(traj_B_fcounts, mdp_world.weights)
            self.halfspace = traj_B_fcounts - traj_A_fcounts
            self.equivalence = True
        elif return_diff > 0:
            #traj A is better
            self.traj_worse = traj_B
            self.traj_better = traj_A
            self.traj_worse_return = np.dot(traj_B_fcounts, mdp_world.weights)
            self.traj_better_return = np.dot(traj_A_fcounts, mdp_world.weights)
            self.halfspace = traj_A_fcounts - traj_B_fcounts
        else:
            #traj B is better
            self.traj_worse = traj_A
            self.traj_better = traj_B
            self.traj_worse_return = np.dot(traj_A_fcounts, mdp_world.weights)
            self.traj_better_return = np.dot(traj_B_fcounts, mdp_world.weights)
            self.halfspace = traj_B_fcounts - traj_A_fcounts
        # print(self.traj_worse)
        # print(self.traj_better)
        # print(self.halfspace)
        # print(self.equivalence)

    
    def __str__(self):
        print_str = "---------- Traj Pair"
        print_str+= "worse trajectory:\n"
        print_str += "{}\n".format(utils.print_traj(self.traj_worse, self.mdp_world, print=False))
        print_str += "return = {}\n\n".format(self.traj_worse_return)
        
        print_str += "better trajectory:\n"
        print_str += "{}\n".format(utils.print_traj(self.traj_better, self.mdp_world, print=False))
        print_str += "return = {}\n\n".format(self.traj_better_return)
        if self.equivalence:
            print_str += "hyperplane!\n"
        else:
            print_str += "halfspace\n"
        print_str += "{}".format(self.halfspace)
        return print_str
        
        