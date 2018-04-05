import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
import numpy as np
import copy as cp
import operator
import random
import math

from collections import Counter

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
    plt.plot(x, y, "ro")
    if fit:
        plt.plot(x, fit_func(x), "b")
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

###############################################################################
# DATA UTILS

def preprocess(df):
    # Delete rows with missing ids
    df_processed = df[((df.id_source_airport != "\\N") 
                      & (df.id_destination_airport != "\\N"))]
    
    # Keep only ["id_source_airport", "id_destination_airport"] col
    # And drop duplicates (corresponding to different airlines)
    df_processed = df_processed[["source_airport", 
                                 "destination_airport"]].drop_duplicates()
    
    return df_processed

###############################################################################
# GEOLOCATION UTILS 

def shoot(lon, lat, azimuth, maxdist=None):
    """Shooter Function. Let us shoot a function. We are going to call it 360 
    times at distance R of the center to plot the circle.
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.
 
    EPS= 0.00000000005
    if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
        alert("Only N-S courses are meaningful, starting at a pole!")
 
    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)
 
    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs (y - c) > EPS):
 
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu
 
    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi    
 
    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)
 
    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi
 
    return (glon2, glat2, baz)

def circle(m, centerlon, centerlat, radius, *args, **kwargs):
    """
    Points à equi distance of the center (in order to plot a circle in the map)
    """
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])
 
    # m.plot(X,Y,**kwargs) #Should work, but doesn"t...
    X,Y = m(X,Y)
    plt.plot(X,Y,**kwargs)
