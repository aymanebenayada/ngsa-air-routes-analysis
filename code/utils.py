from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import copy as cp
import operator
import random
import math

from PyGeoTools.geolocation import GeoLocation

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


###############################################################################
# GEOLOCATION UTILS 

class AffectedAirports(object):

    def __init__(self, airports_info, routes):
        """Delete the routes that have at least one airport within a great-circle
        distance from a center location that produced a spatial hazard. 
        Such airports will be referred to as "affected".
        
        Parameters
        ----------
        airports_info : DataFrame
            Airports information (with at least "latitude" and "longitude" in 
            degrees).
        routes : DataFrame
            Routes with "source_airport", "destination_airport" columns.
        """
        self.airports_info = airports_info
        self.routes = routes

        # Undirected graph representing the routes intially
        arr_edges = np.array(self.routes)
        G = nx.Graph()
        G.add_edges_from(arr_edges)
        self.n_initial_routes = len(G.edges())
        self.n_initial_airports = len(G.nodes())
        self.gcc_size_initial = max(nx.connected_component_subgraphs(G), key=len).number_of_nodes()

        print("Number of initial airports: {}".format(self.n_initial_airports))   
        print("Number of initial routes: {}".format(self.n_initial_routes))
        print("Number of airports in the GCC initially: {}".format(self.gcc_size_initial))

        # New routes DataFrame corresponding to the undirected graph
        self.routes = pd.DataFrame(list(G.edges())).rename(columns={0: "source_airport", 
                                                                    1: "destination_airport"})

    def is_airport_within_dist(self, lat_airport, long_airport, 
                               loc_center, dist_from_center):
        """Check if an airport is within a given distance in kilometers from a
        geographical center (defined by its latitude and longitude in degrees).
        
        Parameters
        ----------
        lat_airport : float
            Latitude in degrees of the airport to consider.
        long_airport : float
            Longitude in degrees of the airport to consider.
        loc_center : GeoLocation
            Center of the hazard
        dist_from_center : float
            Great-circle distance from the center that define the hazard 
            (in kilometers).
        
        Returns
        -------
        TYPE
            Description
        """
        loc_airport = GeoLocation.from_degrees(lat_airport, long_airport)

        if loc_center.distance_to(loc_airport) < dist_from_center:
            return 1
        else:
            return 0

    def get_airports_within_dist(self, lat_center, long_center, dist_from_center):
        """Label every airport given its location inside or outside the "affected"
        area.
        
        Parameters
        ----------
        lat_center : float
            Latitude in degrees of the center location.
        long_center : float
            Longitude in degrees of the center location.
        dist_from_center : float
            Great-circle distance from the center that define the hazard 
            (in kilometers).
        """
        # GeoLocation object of the center of the hazard
        loc_center = GeoLocation.from_degrees(lat_center, long_center)

        self.airports_info["is_affected"] = self.airports_info.apply(lambda row: 
            self.is_airport_within_dist(row["latitude"], row["longitude"], 
                                        loc_center, dist_from_center), 
            axis=1)

        self.airports_to_close = list(self.airports_info[self.airports_info.is_affected==1]["IATA"])

        self.n_airports_to_close = len(self.airports_to_close)
        self.proportion_airports_closed = self.n_airports_to_close/self.n_initial_airports
        print("Proportion of closed airports: {:0.3f}%".format(100*self.proportion_airports_closed))

    def get_new_routes(self, lat_center, long_center, dist_from_center):
        """Label every airport given its location inside or outside the "affected"
        area and update the routes.
        
        Parameters
        ----------
        lat_center : float
            Latitude in degrees of the center location.
        long_center : float
            Longitude in degrees of the center location.
        dist_from_center : float
            Great-circle distance from the center that define the hazard 
            (in kilometers).
        """

        self.get_airports_within_dist(lat_center, long_center, dist_from_center)
        self.new_routes = self.routes[~self.routes.source_airport.isin(self.airports_to_close) 
                                      & ~self.routes.destination_airport.isin(self.airports_to_close)]
        self.n_routes_to_cancel = self.n_initial_routes - self.new_routes.shape[0]
        self.proportion_routes_cancelled = self.n_routes_to_cancel/self.n_initial_routes
        print("Proportion of cancelled routes: {:0.3f}%".format(100*self.proportion_routes_cancelled))

        # Size of the GCC of the new network 
        arr_edges = np.array(self.new_routes)
        G = nx.Graph()
        G.add_edges_from(arr_edges)
        self.gcc_size_new = max(nx.connected_component_subgraphs(G), key=len).number_of_nodes()

        # Size of the GCC of the new network divided by the size of the GCC
        # of the intial network
        self.proportion_GCC = self.gcc_size_new/self.gcc_size_initial
        print("new GCC size / initial GCC size: {:0.3f}%".format(100*self.proportion_GCC))

