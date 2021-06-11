"""A variety of prebuild worlds for debugging and testing"""
import src.mdp as mdp
import numpy as np


def create_aaai19_toy_world():
    #features is a 2-d array of tuples
    num_rows = 2
    num_cols = 3
    features =[[(1, 0), (0, 1), (1, 0)],
            [(1, 0), (1, 0), (1, 0)]]
    weights = [-1,-4]
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols)] #states indexed by row and then column
    #print(initials)
    terminals = [(0,0)]
    gamma = 0.9
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world



def create_safety_island_world():
    #Taken from the AI safety grid worlds paper
    #features is a 2-d array of tuples
    num_rows = 6
    num_cols = 8
    wall = None
    goal = (1,0,0)
    white = (0,1,0)
    water = (0,0,1)
    features = [[water, water, wall, wall, wall, wall, wall, wall],
                [water, water, white, white, white, white, white, water],
                [water, water, white, white, white, white, white, water],
                [water, white, white, white, white, white, white, water],
                [water, white, white, goal, white, white, water, water],
                [water, wall, wall, wall, wall, wall, wall, wall]]
    weights = [+50, -1, -50] #goal, movement, water
    #can't start in water or wall
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if (features[r][c] != None and features[r][c] != water)] #states indexed by row and then column
    print(initials)
    terminals = [(4,3)]
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world


def create_safety_island_world_nowalls():
    #Taken from the AI safety grid worlds paper
    #features is a 2-d array of tuples
    num_rows = 3
    num_cols = 8
    wall = None
    goal = (1,0,0)
    white = (0,1,0)
    water = (0,0,1)
    features = [[water, water, white, white, white, white, white, water],
                [water, white, white, white, white, white, white, water],
                [water, white, white, goal, white, white, water, water]]
    weights = [+50, -1, -50] #goal, movement, water
    #can't start in water or wall
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if (features[r][c] != None and features[r][c] != water)] #states indexed by row and then column
    print(initials)
    terminals = [(2,3)]
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world


def create_safety_lava_world():
    #Taken from the AI safety grid worlds paper
    #features is a 2-d array of tuples
    num_rows = 7
    num_cols = 9
    wall = None
    goal = (1,0,0)
    white = (0,1,0)
    lava = (0,0,1)
    features = [[wall, wall, wall, wall, wall, wall, wall, wall, wall],
                [wall, white, white, lava, lava, lava, white, goal, wall],
                [wall, white, white, lava, lava, lava, white, white, wall],
                [wall, white, white, white, white, white, white, white, wall],
                [wall, white, white, white, white, white, white, white, wall],
                [wall, white, white, white, white, white, white, white, wall],
                [wall, wall, wall, wall, wall, wall, wall, wall, wall]]
    weights = [+50, -1, -50] #goal, movement, lava
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if features[r][c] != None] #states indexed by row and then column
    print(initials)
    terminals = [(1,7)]
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world


def create_safety_lava_world_nowalls():
    #Taken from the AI safety grid worlds paper
    #features is a 2-d array of tuples
    num_rows = 3
    num_cols = 7
    wall = None
    goal = (1,0,0)
    white = (0,1,0)
    lava = (0,0,1)
    features = [[white, white, lava, lava, lava, white, goal],
                [white, white, lava, lava, lava, white, white],
                [white, white, white, white, white, white, white]]
    weights = [+50, -1, -50] #goal, movement, lava
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if features[r][c] != None] #states indexed by row and then column
    print(initials)
    terminals = [(0,6)]
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world


def create_cakmak_task1():
    #features is a 2-d array of tuples
    num_rows = 6
    num_cols = 7
    wall = None
    star = (1,0,0)
    white = (0,1,0)
    gray = (0,0,1)
    features = [[wall, star, wall, wall, white, wall, wall],
                [wall, white, white, white, white, white, wall],
                [wall, gray, wall, wall, wall, white, wall],
                [wall, gray, wall, wall, wall, white, wall],
                [wall, white, white, white, white, white, wall],
                [white, white, wall, wall, wall, wall, wall]]
    weights = [2,-1,-1]
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if features[r][c] != None] #states indexed by row and then column
    print(initials)
    terminals = [(0,1)]
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world

def create_cakmak_task2():
    world = create_cakmak_task1()
    world.weights = [2, -1, -10]
    return world


def create_cakmak_task3():
    #features is a 2-d array of tuples
    num_rows = 6
    num_cols = 6
    wall = None
    star = (1,0,0)
    diamond = (0,1,0)
    white = (0,0,1)
    features = [[star, wall, white, white, wall, diamond],
                [white, wall, white, white, wall, white],
                [white, white, white, white, white, white],
                [white, white, white, white, white, white],
                [white, white, white, white, white, white],
                [white, white, white, white, white, white]]
    weights = [1,1,-1]
    initials = [(r,c) for r in range(num_rows) for c in range(num_cols) if features[r][c] != None] #states indexed by row and then column
    print(initials)
    terminals = [(0,0), (0,5)] 
    gamma = 0.95
    world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)
    return world

def create_cakmak_task4():
    #note that Cakmak and Lopes appear to be using gamma = 1, but in this case you don't get the teaching set they show...
    #increasing the reward of the diamond feature gives a comparable result to their paper.
    world = create_cakmak_task3()
    world.weights = [1,3,-1]
    return world