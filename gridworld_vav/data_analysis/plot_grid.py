import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import sys


def plot_dashed_arrow(state, width, ax, direction, arrow_color='k'):
    print("plotting dashed arrow", direction)
    h_length = 0.15
    shaft_length = 0.4
    
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=None, head_length=None, fc=arrow_color, ec=arrow_color,linewidth=4, linestyle=':',fill=False) 
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = h_length
        y_coord += shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -h_length
        y_coord += -shaft_length + h_length
    elif direction is 'left':
        x_end = -h_length
        y_end = 0
        x_coord += -shaft_length + h_length
    elif direction is 'right':
        x_end = h_length
        y_end = 0
        x_coord += shaft_length - h_length
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length,  fc=arrow_color, ec=arrow_color,linewidth=4, fill=False,length_includes_head = True) 

def plot_arrow(state, width, ax, direction, arrow_color='k'):
    print("plotting arrow", direction)
    h_length = 0.15
    shaft_length = 0.4
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc=arrow_color, ec=arrow_color,linewidth=4) 

def plot_dot(state, width, ax):
    ax.plot(state % width, state // width, 'ko',markersize=10)
    
def plot_questionmark(state, width, ax):
    ax.plot(state % width, state // width, 'k', marker=r'$?$',markersize=40)
    

def plot_optimal_policy(pi, feature_mat):
    plt.figure()

    ax = plt.axes() 
    count = 0
    print(pi)
    rows,cols = len(pi), len(pi[0])
    for line in pi:
        for el in line:
            print("optimal action", el)
            # could be a stochastic policy with more than one optimal action
            for char in el:
                print(char)
                if char is "^" or char == (-1,0):
                    plot_arrow(count, cols, ax, "up")
                elif char is "v" or char == (1,0): 
                    plot_arrow(count, cols, ax, "down")
                elif char is ">" or char == (0,1):
                    plot_arrow(count, cols, ax, "right")
                elif char is "<" or char == (0,-1):
                    plot_arrow(count, cols, ax, "left")
                elif char is ".":
                    plot_dot(count, cols, ax)
                elif el is "w":
                    #wall
                    pass
                else:
                    print("error in policy format")
                    sys.exit()
            count += 1

    
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    cmap = colors.ListedColormap(['black','white','tab:red', 'tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.show()

def plot_optimal_policy_vav(pi, feature_mat, walls=False, filename=False, show=False, arrow_color='k', feature_colors = None):
    #takes a dictionary of policy optimal actions
    #takes a 2d array of feature vectors
    plt.figure()

    ax = plt.axes() 
    count = 0
    print(pi)
    rows,cols = len(feature_mat), len(feature_mat[0])
    for r in range(rows):
        for c in range(cols):
            if feature_mat[r][c]:
                opt_actions = pi[(r,c)]
                for a in opt_actions:
                    print("optimal action", a)
                    # could be a stochastic policy with more than one optimal action
                    if a is None:
                        plot_dot(count, cols, ax)
                    else:
                        if a == (-1,0):
                            plot_arrow(count, cols, ax, "up", arrow_color)
                        elif a == (1,0): 
                            plot_arrow(count, cols, ax, "down", arrow_color)
                        elif a == (0,1):
                            plot_arrow(count, cols, ax, "right", arrow_color)
                        elif a == (0,-1):
                            plot_arrow(count, cols, ax, "left", arrow_color)
                        elif a is None:
                            plot_dot(count, cols, ax)
                        elif a is "w":
                            #wall
                            pass
                        else:
                            print("error in policy format")
                            #sys.exit()
            count += 1

    print(feature_mat)
    
    #use for wall states
    #if walls:
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    
    #mat =[[0,0],[2,2]]
    feature_set = set()
    for mrow in mat:
        for m in mrow:
            feature_set.add(m)
    num_features = len(feature_set)
    print(mat)
    if feature_colors is None:
        all_colors = ['black','white','tab:red','tab:blue','tab:gray','tab:green','tab:purple', 'tab:orange',  'tab:cyan']
    else:
        all_colors = feature_colors
    colors_to_use = []
    for f in range(9):#hard coded to only have 9 features right now
        if f in feature_set:
            colors_to_use.append(all_colors[f])
    cmap = colors.ListedColormap(colors_to_use)
    # else:
    #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
    #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    
    #input()
    
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    elif show:
        plt.show()


def plot_test_questions(question_list, feature_mat, walls=False, filename=False, show=False, arrow_color='k', feature_colors = None):
    #takes a dictionary of policy optimal actions
    #takes a 2d array of feature vectors
    plt.figure()

    ax = plt.axes() 
    count = 0
    
    rows,cols = len(feature_mat), len(feature_mat[0])
    for r in range(rows):
        for c in range(cols):
            if feature_mat[r][c]:
                for (s,a) in question_list:
                    if s == (r,c):
                        if type(a) is list:
                            opt_actions = a
                        else:
                            opt_actions = [a]
                        for a in opt_actions:
                            print("optimal action", a)
                            # could be a stochastic policy with more than one optimal action
                            if a is None:
                                #plot_dot(count, cols, ax)
                                continue # don't plot anything at terminal no choice there anyways
                            else:
                                # if a == (-1,0):
                                #     plot_arrow(count, cols, ax, "up", arrow_color)
                                # elif a == (1,0): 
                                #     plot_arrow(count, cols, ax, "down", arrow_color)
                                # elif a == (0,1):
                                #     plot_arrow(count, cols, ax, "right", arrow_color)
                                # elif a == (0,-1):
                                #     plot_arrow(count, cols, ax, "left", arrow_color)
                                # elif a is None:
                                plot_questionmark(count, cols, ax)
                                # elif a is "w":
                                #     #wall
                                #     pass
                                # else:
                                #     print("error in policy format")
                                #     #sys.exit()
            count += 1

    print(feature_mat)
    
    #use for wall states
    #if walls:
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    
    #mat =[[0,0],[2,2]]
    feature_set = set()
    for mrow in mat:
        for m in mrow:
            feature_set.add(m)
    num_features = len(feature_set)
    print(mat)
    if feature_colors is None:
        all_colors = ['black','white','tab:red','tab:blue','tab:gray','tab:green','tab:purple', 'tab:orange',  'tab:cyan']
    else:
        all_colors = feature_colors
    colors_to_use = []
    for f in range(9):#hard coded to only have 9 features right now
        if f in feature_set:
            colors_to_use.append(all_colors[f])
    cmap = colors.ListedColormap(colors_to_use)
    # else:
    #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
    #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    
    #input()
    
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()



def plot_preference_query(good_traj, bad_traj, feature_mat, walls=False, filename=False, show=False, 
                    good_arrow_color='b', bad_arrow_color='r', feature_colors = None):
    #Takes in two trajs good and bad and plots good in solid and bad in dotted
    
    plt.figure()
    ax = plt.axes() 
    count = 0
    rows,cols = len(feature_mat), len(feature_mat[0])
    #plot good trajectory
    arrow_color=good_arrow_color
    for r in range(rows):
        for c in range(cols):
            if feature_mat[r][c]:
                for (s,a) in good_traj:
                    if s == (r,c):
                        if type(a) is list:
                            opt_actions = a
                        else:
                            opt_actions = [a]
                        for a in opt_actions:
                            print("optimal action", a)
                            # could be a stochastic policy with more than one optimal action
                            if a is None:
                                plot_dot(count, cols, ax)
                            else:
                                if a == (-1,0):
                                    plot_arrow(count, cols, ax, "up", arrow_color)
                                elif a == (1,0): 
                                    plot_arrow(count, cols, ax, "down", arrow_color)
                                elif a == (0,1):
                                    plot_arrow(count, cols, ax, "right", arrow_color)
                                elif a == (0,-1):
                                    plot_arrow(count, cols, ax, "left", arrow_color)
                                elif a is None:
                                    plot_dot(count, cols, ax)
                                elif a is "w":
                                    #wall
                                    pass
                                else:
                                    print("error in policy format")
                                    #sys.exit()
            count += 1

    #plot bad trajectory
    arrow_color=bad_arrow_color
    count = 0
    for r in range(rows):
        for c in range(cols):
            if feature_mat[r][c]:
                for (s,a) in bad_traj:
                    if s == (r,c):
                        if type(a) is list:
                            opt_actions = a
                        else:
                            opt_actions = [a]
                        for a in opt_actions:
                            print("optimal action", a)
                            # could be a stochastic policy with more than one optimal action
                            if a is None:
                                plot_dot(count, cols, ax)
                            else:
                                if a == (-1,0):
                                    plot_dashed_arrow(count, cols, ax, "up", arrow_color)
                                elif a == (1,0): 
                                    plot_dashed_arrow(count, cols, ax, "down", arrow_color)
                                elif a == (0,1):
                                    plot_dashed_arrow(count, cols, ax, "right", arrow_color)
                                elif a == (0,-1):
                                    plot_dashed_arrow(count, cols, ax, "left", arrow_color)
                                elif a is None:
                                    plot_dot(count, cols, ax)
                                elif a is "w":
                                    #wall
                                    pass
                                else:
                                    print("error in policy format")
                                    #sys.exit()
            count += 1

   
    
    #use for wall states
    #if walls:
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    
    #mat =[[0,0],[2,2]]
    feature_set = set()
    for mrow in mat:
        for m in mrow:
            feature_set.add(m)
    num_features = len(feature_set)
    print(mat)
    if feature_colors is None:
        all_colors = ['black','white','tab:red','tab:blue','tab:gray','tab:green','tab:purple', 'tab:orange',  'tab:cyan']
    else:
        all_colors = feature_colors
    colors_to_use = []
    for f in range(9):#hard coded to only have 9 features right now
        if f in feature_set:
            colors_to_use.append(all_colors[f])
    cmap = colors.ListedColormap(colors_to_use)
    # else:
    #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
    #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    
    #input()
    
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()


def plot_optimal_policy_vav_grid(pis, feature_mats, g_rows, g_cols, walls=False, filename=False):
    #size is tuple for rows / cols of
    #takes a dictionary of policy optimal actions
    #takes a 2d array of feature vectors
    fig, axs = plt.subplots(g_rows, g_cols)
    cnt = 0
    for ax in axs:#r in range(g_rows):
        #for c in range(g_cols):
            #print(r,c)
            #ax = axs[r,c]
            #ax.set_title('Axis [0,0]')
            pi = pis[cnt]
            feature_mat = feature_mats[cnt]
            cnt += 1
            count = 0
            #print(pi)
            rows,cols = len(feature_mat), len(feature_mat[0])
            for r in range(rows):
                for c in range(cols):
                    opt_actions = pi[(r,c)]
                    for a in opt_actions:
             #           print("optimal action", a)
                        # could be a stochastic policy with more than one optimal action
                        if a is None:
                            plot_dot(count, cols, ax)
                        else:
                            if a == (-1,0):
                                plot_arrow(count, cols, ax, "up")
                            elif a == (1,0): 
                                plot_arrow(count, cols, ax, "down")
                            elif a == (0,1):
                                plot_arrow(count, cols, ax, "right")
                            elif a == (0,-1):
                                plot_arrow(count, cols, ax, "left")
                            elif a is None:
                                plot_dot(count, cols, ax)
                            elif a is "w":
                                #wall
                                pass
                            else:
                                print("error in policy format")
                                sys.exit()
                    count += 1

            # print(feature_mat)
            
            #use for wall states
            #if walls:
            mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
            
            #mat =[[0,0],[2,2]]
            feature_set = set()
            for mrow in mat:
                for m in mrow:
                    feature_set.add(m)
            num_features = len(feature_set)
            # print(mat)
            all_colors = ['black','white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan']
            colors_to_use = []
            for f in range(9):#hard coded to only have 9 features right now
                if f in feature_set:
                    colors_to_use.append(all_colors[f])
            cmap = colors.ListedColormap(colors_to_use)
            # else:
            #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
            #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
            
            #input()
            
            #convert feature_mat into colors
            #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
            
            ax.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

            #ax = plt.gca()

            ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
            ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
            #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(plt.NullLocator())
            #cbar = plt.colorbar(heatmap)
            #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    
def plot_test_query(state, better_action, worse_action, feature_mat, equal_pref = False):

    plt.figure()
    ax = plt.axes() 
    count = 0
    rows,cols = len(feature_mat), len(feature_mat[0])
    if better_action is "^":
        plot_arrow(state, cols, ax, "up")
    elif better_action is "v":
        plot_arrow(state, cols, ax, "down")
    elif better_action is ">":
        plot_arrow(state, cols, ax, "right")
    elif better_action is "<":
        plot_arrow(state, cols, ax, "left")
        
    if equal_pref:
        if worse_action is "^":
            plot_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_arrow(state, cols, ax, "left")

    
    else:
    
        if worse_action is "^":
            plot_dashed_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_dashed_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_dashed_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_dashed_arrow(state, cols, ax, "left")

    
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    cmap = colors.ListedColormap(['black','white','tab:blue','tab:red','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')
    # Add the grid
    ax = plt.gca()
    # Minor ticks
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    #remove ticks
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off') # labels along the bottom edge are off

    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.show()
    
    
if __name__=="__main__":
    pi = [['v', '^><','.'],['<>v','<','>'],['<>^v','v' ,'^']]
    feature_mat = [[(1,0,0),(0,1,0),(0,0,1)],[(0,0,0,1),(0,0,0,0,1),(0,0,0,0,0,1)],[(0,0,0,0,0,0,1), (0,0,0,0,0,0,0,1),None]  ]      
    plot_optimal_policy(pi, feature_mat)
    
    state = 3  #the integer value of state starting from top left and reading left to right, top to bottom.
    better_action = "v"
    worse_action = "<"
    #plot the optimal test query, where the right answer is bolded  (add equal_pref=True argument if both are equally good)
    plot_test_query(state, better_action, worse_action, feature_mat)
    
    state = 4  #the integer value of state starting from top left and reading left to right, top to bottom.
    better_action = "v"
    worse_action = "<"
    plot_test_query(state, better_action, worse_action, feature_mat, equal_pref = True)

