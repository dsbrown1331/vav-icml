import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os
import sys
exp_path = os.path.dirname(os.path.abspath(__file__))
print(exp_path)
project_path = os.path.abspath(os.path.join(exp_path, ".."))
print(project_path)
# sys.path.insert(0, project_path)
# print(sys.path)



num_rows_list = [4,5,6,7,8]#[4,8,16]
num_features_list = [3,4,5,6,7,8]



verifier_list = [ "arp-w","arp-pref","scot", "arp-bb","state-value-critical-0.2"]
name_map = {"arp-pref":"ARP-pref", "scot":"SCOT", "arp-bb":"ARP-bb", "arp-w":"ARP-w", "state-value-critical-0.2":"CS-0.2"}
exp_data_dir = os.path.join(project_path, "results/arp_gaussian")

color_lines = ['b^-', 'gs-.', 'ro--', 'kx:','cd--','y+--','m^:']

plt.figure()

for i,v in enumerate(verifier_list):
    plt.plot([], color_lines[i], linewidth=3, label=name_map[v])
plt.xticks(num_rows_list,fontsize=15) 
plt.yticks(fontsize=15) 
plt.xlabel('Grid world width', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.legend(loc='best', fontsize=20, ncol=5)
plt.axis('off')
#plt.tight_layout()
# plt.show()
    

##first let's plot the number of states along the x-axis and accuracy and test size on y-axis for the different methods

#go through and calculate the means for each of these 
for num_features in [5]:

    all_accuracies = {}
    all_test_sizes = {}
    for v in verifier_list:
        all_accuracies[v] = []
        all_test_sizes[v] = []
    for num_rows in num_rows_list:
        num_cols = num_rows #keep it square grid for  now
        for i, verifier_name in enumerate(verifier_list):
            filename = "arp{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            print("reading from", filename)
            my_data = genfromtxt(os.path.join(exp_data_dir, filename), delimiter=',')
            #columns are num_correct, num_tested, size_test_set
            num_correct = my_data[:,0]
            num_tested = my_data[:,1]
            test_sizes = my_data[:,2]
            ave_accuracy = np.mean(num_correct / num_tested)
            all_accuracies[verifier_name].append(ave_accuracy)
            if verifier_name == "ranking-halfspace":
                all_test_sizes[verifier_name].append(1)
            else:
                all_test_sizes[verifier_name].append(np.mean(test_sizes))

    print(all_accuracies)
    print(all_test_sizes)

    #make plot of accuracies
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.title("Num Features = {}".format(num_features), fontsize=25)
    for i,v in enumerate(verifier_list):
        plt.plot(num_rows_list, all_accuracies[v], color_lines[i], linewidth=3, label=name_map[v])
    plt.xticks(num_rows_list,fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.xlabel('Grid World Width', fontsize=25)
    plt.ylabel('Accuracy', fontsize=25)
    plt.legend(loc='lower right', fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./data_analysis/figs/arp_gauss_basic_features{}_accuracy_nolegend.png'.format(num_features))


    #make plot of test sizes
    plt.figure()
    plt.title("Num Features = {}".format(num_features), fontsize=25)
    ax = plt.subplot(1,1,1)
    for i,v in enumerate(verifier_list):
        ax.semilogy(num_rows_list, all_test_sizes[v], color_lines[i], linewidth=3, label=name_map[v])
    #ax.tick_params(bottom=False, top=False, left=True, right=True)
    #ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    plt.xticks(num_rows_list,fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.xlabel('Grid World Width', fontsize=25)
    plt.ylabel('Test Queries', fontsize=25)
    #plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/arp_gauss_basic_features{}_queries_nolegend.png'.format(num_features))

    # plt.show()


#verifier_list = ["ranking-halfspace", "scot", "optimal_action", "state-value-critical-0.1"]


##now let's look at number of features along the xaxis for a could different grid sizes
#go through and calculate the means for each of these 
for num_rows in [8]:

    all_accuracies = {}
    all_test_sizes = {}
    for v in verifier_list:
        all_accuracies[v] = []
        all_test_sizes[v] = []
    for num_features in num_features_list:
        num_cols = num_rows #keep it square grid for  now
        for i, verifier_name in enumerate(verifier_list):
            filename = "arp{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            print("reading from", filename)
            my_data = genfromtxt(os.path.join(exp_data_dir, filename), delimiter=',')
            #columns are num_correct, num_tested, size_test_set
            num_correct = my_data[:,0]
            num_tested = my_data[:,1]
            test_sizes = my_data[:,2]
            ave_accuracy = np.mean(num_correct / num_tested)
            all_accuracies[verifier_name].append(ave_accuracy)
            if verifier_name == "ranking-halfspace":
                all_test_sizes[verifier_name].append(1)
            else:
                all_test_sizes[verifier_name].append(np.mean(test_sizes))

    print(all_accuracies)
    print(all_test_sizes)

    #make plot of accuracies
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.title("Grid World Width = {}".format(num_rows), fontsize=25)
    for i,v in enumerate(verifier_list):
        plt.plot(num_features_list, all_accuracies[v], color_lines[i], linewidth=3, label=name_map[v])
    plt.xticks(num_features_list,fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.xlabel('Number of Reward Features', fontsize=25)
    plt.ylabel('Accuracy', fontsize=25)
    plt.legend(loc='lower right', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/arp_gauss_basic_size{}_accuracy_nolegend.png'.format(num_rows))
    # plt.show()


    #make plot of test sizes
    plt.figure()
    plt.title("Grid World Width = {}".format(num_rows), fontsize=25)
    ax = plt.subplot(1,1,1)
    for i,v in enumerate(verifier_list):
        ax.semilogy(num_features_list, all_test_sizes[v], color_lines[i], linewidth=3, label=name_map[v])
    #ax.tick_params(bottom=False, top=False, left=True, right=True)
    #ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    plt.xticks(num_features_list,fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.xlabel('Number of Reward Features', fontsize=25)
    plt.ylabel('Test Queries', fontsize=25)
    #plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/arp_gauss_basic_size{}_queries_nolegend.png'.format(num_rows))

    plt.show()