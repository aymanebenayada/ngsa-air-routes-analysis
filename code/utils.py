from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy as cp
import operator
import random


def find_key(d, value):
    result = []
    for key in d:
        if d[key] == value:
            result.append(key)
    return result


def plot_distribution(G, what="in", xscale="log", yscale="log", 
                      xlabel="Degree", title="Degree distribution", fit=False):
    # Get degree per node
    if what=="in":
        per_node = dict(G.in_degree())
    elif what=="out":
        per_node = dict(G.out_degree())
    
    # (Degree, #Occurences) dict
    count = Counter(per_node.values())

    x = list(count.keys())
    y = list(count.values())
    
    if fit:
        log_x = np.log(x)
        log_y = np.log(y)
        coeff = np.polyfit(log_x,log_y,1)
        log_C = coeff[1]
        alpha = -coeff[0]
        fit_func = lambda k: exp(log_C)*k**-alpha
    
    # Plot
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.plot(x, y, 'ro')
    if fit:
        plt.plot(x, fit_func(x), 'b')
        plt.figtext(0.75, 0.75, r"$\alpha = {:0.2f}$".format(alpha))
        plt.figtext(0.75, 0.70, r"$C = {:0.2f}$".format(exp(log_C)))
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid()
    plt.show()

def random_removal_and_size_gcc_rest(prop_to_rmv, G):
    G_rmvd = cp.deepcopy(G)
    node_list = G_rmvd.nodes()
    n_nodes = len(node_list)
    nodes_to_rmv = random.sample(node_list, k=int(prop_to_rmv*n_nodes))
    G_rmvd.remove_nodes_from(nodes_to_rmv)
    n_nodes_rmvd = G_rmvd.number_of_nodes()
    Gc = max(nx.connected_component_subgraphs(G_rmvd), key=len)
    n_nodes_gc = Gc.number_of_nodes()
    return [n_nodes_gc, n_nodes_rmvd-n_nodes_gc]

def targetted_removal_and_size_gcc_rest(prop_to_rmv, G):
    G_rmvd = cp.deepcopy(G)
    sorted_nodes_by_degree = sorted(dict(G_rmvd.degree()).items(),
                                    key=operator.itemgetter(1), reverse=True)
    n_nodes_to_rmv = int(G_rmvd.number_of_nodes()*prop_to_rmv)
    nodes_to_rmv = [key_value[0] for key_value in sorted_nodes_by_degree[:n_nodes_to_rmv]]
    G_rmvd.remove_nodes_from(nodes_to_rmv)
    n_nodes_rmvd = G_rmvd.number_of_nodes()
    Gc = max(nx.connected_component_subgraphs(G_rmvd), key=len)
    n_nodes_gc = Gc.number_of_nodes()
    return [n_nodes_gc, n_nodes_rmvd-n_nodes_gc]

def create_features_for_plot(prop_min, prop_max, n_points, G):
    props = np.linspace(prop_min, prop_max, num=n_points)
    Y_random = []
    Y_target = []
    for prop in props:
        Y_random.append(random_removal_and_size_gcc_rest(prop, G))
        Y_target.append(targetted_removal_and_size_gcc_rest(prop, G))
    Y_random = np.array(Y_random)
    Y_target = np.array(Y_target)
    return (props, Y_random, Y_target)

def plot_robustness(props, Y_random, Y_target, 
                    title="Robustness to Random Failures and to Targetted Attacks"):
    plt.plot(props, Y_random[:,0])
    plt.plot(props, Y_random[:,1])
    plt.plot(props, Y_target[:,0])
    plt.plot(props, Y_target[:,1])
    plt.title(title)
    plt.grid()
    plt.xlabel("Proportion of nodes removed")
    plt.ylabel("Sizes")
    plt.legend(["Random GCC","Random Rest", "Target GCC", "Target Rest"])
    plt.show()

