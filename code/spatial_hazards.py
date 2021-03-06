#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class helps simulating a hazard against an air traffic network.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
import numpy as np
import random

from PyGeoTools.geolocation import GeoLocation
from mpl_toolkits.basemap import Basemap
from matplotlib.pyplot import savefig
from utils import circle

class SpatialHazards(object):

    def __init__(self, airports_info, routes, avg_shortest_path=False):
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
        avg_shortest_path : bool, optional
            If True compute the average shortest path of the initial network
            and the affected one (average shortest path across the different 
            subgraphs, weighted by the number of nodes of the subgraph).
        """
        self.airports_info = airports_info
        self.routes = routes
        self.avg_shortest_path = avg_shortest_path

        # Undirected graph representing the routes intially
        arr_edges = np.array(self.routes)
        G = nx.Graph()
        G.add_edges_from(arr_edges)
        self.n_initial_routes = len(G.edges())
        connected_components = list(nx.connected_component_subgraphs(G))
        self.n_initial_airports = len(G.nodes())
        self.gcc_initial = max(connected_components, key=len)
        self.gcc_size_initial = self.gcc_initial.number_of_nodes()
        self.robust_points = []
        self.unrobust_points = []
        self.robust_points_ts = 0.3
        self.unrobust_points_ts = -0.05


        print("Number of initial airports: {}".format(self.n_initial_airports))   
        print("Number of initial routes: {}".format(self.n_initial_routes))
        print("Number of airports in the GCC initially: {}"\
             .format(self.gcc_size_initial))

        if self.avg_shortest_path:

            # Average shortest path across the different subgraphs, weighted by 
            # the number of nodes of the subgraph
            self.avg_shortest_path_initial = 0
            for graph in connected_components:
                self.avg_shortest_path_initial += nx.average_shortest_path_length(graph)\
                                                  *graph.number_of_nodes()
            self.avg_shortest_path_initial /= G.number_of_nodes()

            print("Average shortest path initially: {:0.2f}".format(
                self.avg_shortest_path_initial))

        # New routes DataFrame corresponding to the undirected graph
        self.routes = pd.DataFrame(list(G.edges())).rename(
            columns={0: "source_airport", 1: "destination_airport"})

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
        int
            1 if the airport is inside the circular area defined by the 
            center and the radius.
        """
        loc_airport = GeoLocation.from_degrees(lat_airport, long_airport)

        # If the center and the airport are the same
        try:
            if loc_center.distance_to(loc_airport) < dist_from_center:
                return 1
            else:
                return 0
        except ValueError:
            return 1

    def get_airports_within_dist(self, lat_center, long_center, dist_from_center, 
                                 verbose=False):
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
        verbose : bool, optional
            If True, print useful information.
        
        Returns
        -------
        list(str)
            List of the closed airports.
        """
        # GeoLocation object of the center of the hazard
        loc_center = GeoLocation.from_degrees(lat_center, long_center)

        self.airports_info["is_affected"] = self.airports_info.apply(lambda row: 
            self.is_airport_within_dist(row["latitude"], row["longitude"], 
                                        loc_center, dist_from_center), 
            axis=1)

        self.airports_to_close = list(self.airports_info[
            self.airports_info.is_affected==1]["IATA"])

        self.n_airports_to_close = len(self.airports_to_close)
        self.proportion_airports_closed = self.n_airports_to_close\
                                          /self.n_initial_airports
        if verbose:
            print("Proportion of closed airports: {:0.3f}%".format(
                100*self.proportion_airports_closed))
            
        return self.airports_to_close

    def get_new_routes(self, lat_center, long_center, dist_from_center, 
                       verbose=False):
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
        verbose : bool, optional
            If True, print useful information.
        
        Returns
        -------
        tuple
        """
        self.get_airports_within_dist(lat_center, long_center, dist_from_center, verbose=verbose)
        
        self.new_routes = self.routes[~self.routes.source_airport.isin(self.airports_to_close) 
                                      & ~self.routes.destination_airport.isin(self.airports_to_close)]

        self.closed_routes = self.routes[self.routes.source_airport.isin(self.airports_to_close) 
                                         | self.routes.destination_airport.isin(self.airports_to_close)]

        self.n_routes_to_cancel = self.n_initial_routes - self.new_routes.shape[0]

        self.proportion_routes_cancelled = self.n_routes_to_cancel/self.n_initial_routes

        # Size of the GCC of the new network (and average shortest path)
        arr_edges = np.array(self.new_routes)
        G = nx.Graph()
        G.add_edges_from(arr_edges)

        if self.proportion_routes_cancelled - self.proportion_airports_closed < self.unrobust_points_ts:
            self.unrobust_points.append((lat_center, long_center, dist_from_center))

        if self.proportion_routes_cancelled - self.proportion_airports_closed > self.robust_points_ts:
            self.robust_points.append((lat_center, long_center, dist_from_center))

        try:
            connected_components = list(nx.connected_component_subgraphs(G))
            self.gcc_new = max(connected_components, key=len)
            self.gcc_size_new = self.gcc_new.number_of_nodes()


            # Size of the GCC of the new network divided by the size of the size of 
            # GCC of the intial network (and average shortest path)
            self.proportion_gcc_size = self.gcc_size_new/self.gcc_size_initial

            if self.avg_shortest_path: 

                self.avg_shortest_path_new = 0
                for graph in connected_components:
                    self.avg_shortest_path_new += nx.average_shortest_path_length(graph)\
                                                  *graph.number_of_nodes()
                self.avg_shortest_path_new /= G.number_of_nodes()

                # Average shortest path of the initial GCC divided by the average 
                # shortest path of the new GCC (the closer to 1, the less affected
                # the network)
                self.proportion_avg_shortest_path = self.avg_shortest_path_initial\
                                                        /self.avg_shortest_path_new
        except:
            self.proportion_gcc_size = 0

            if self.avg_shortest_path:
                self.proportion_avg_shortest_path = 0

        if verbose:

            print("Proportion of cancelled routes: {:0.3f}%".format(
                100*self.proportion_routes_cancelled))
            print("new GCC size / initial GCC size: {:0.3f}%".format(
                100*self.proportion_gcc_size))

            if self.avg_shortest_path:
                print("initial average shortest path / new: {:0.3f}%".format(
                    100*self.proportion_avg_shortest_path))
        
        if self.avg_shortest_path:
            return self.proportion_airports_closed, self.proportion_routes_cancelled,\
                   self.proportion_gcc_size, self.proportion_avg_shortest_path
        else:
            return self.proportion_airports_closed, self.proportion_routes_cancelled,\
                   self.proportion_gcc_size

    def simulate_hazard(self, lat=None, lng=None, rad=None, region="Europe", 
                        verbose=False):
        """Simulate a hazard with a center location and a radius (great-circle
        distance from the center location in km).
        
        Parameters
        ----------
        lat : None, optional
            Latitude of the center location of the hazard, if None a random center 
            location with a random radius will be chosen (constrained to Europe).
        lng : None, optional
            Longitude of the center location of the hazard, if None a random center 
            location with a random radius will be chosen (constrained to Europe).
        rad : None, optional
            Radius the hazard (great-circle distance in km), if None a random center 
            location with a random radius will be chosen (constrained to Europe).
        region : str, optional
            Geographical region where to simulate the hazards.
        verbose : bool, optional
            If True, print useful information.
        
        Returns
        -------
        tuple
            Proportion of airports closed, proportion of routes cancelled,
            size of the GCC of the new network divided by the size of the GCC
            of the initial network
        """
        if lat==None or lng==None or rad==None:
            if region == 'Europe':               
                            LIM_WEST = -9
                            LIM_EST = 27
                            LIM_SOUTH = 35
                            LIM_NORTH = 72
                            LIM_RADIUS = 5000
                        
            elif region == 'US':
                LIM_WEST = -123
                LIM_EST = -76
                LIM_SOUTH = 31
                LIM_NORTH = 44
                LIM_RADIUS = 5000

            lat_ = random.uniform(LIM_SOUTH, LIM_NORTH)
            lng_ = random.uniform(LIM_WEST, LIM_EST)
            rad_ = random.uniform(1, LIM_RADIUS)

        else:
            lat_ = lat
            lng_ = lng
            rad_ = rad

        self.lat_center = lat_
        self.long_center = lng_
        self.dist_from_center = rad_

        return self.get_new_routes(lat_, lng_, rad_, verbose)

    def plot_hazard(self, title=None, n=None, iceland=False, 
                    operational_routes_color="mediumblue",
                    closed_routes_color="firebrick", save_fig=None):
        """Used after simulating a hazard (with self.simulate_hazard).
        
        Parameters
        ----------
        title : None, optional
            Title of the plot.
        n : None, optional
            Number of maximum routes (closed and operational respectively) 
            to show. 
        iceland : bool, optional
            If True, plot a larger map showing Iceland.
        operational_routes_color : str, optional
            Color for the routes that are not closed.
        closed_routes_color : str, optional
            Color for the routes that have been closed by the hazard.
        save_fig : bool, optional
            Specify a suffix for the figure to be saved.
        """
    
        # Settle the environment to plot

        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.size"] = 10.
        mpl.rcParams["axes.labelsize"] = 8.
        mpl.rcParams["xtick.labelsize"] = 6.
        mpl.rcParams["ytick.labelsize"] = 6.
        fig = plt.figure(figsize=(16, 9))

        if title is not None:
            title = "\n".join((title, "Closed airports: {:0.1f}% - Closed routes: {:0.1f}% - MCS: {:0.1f}%"\
                              .format(100*self.proportion_airports_closed, 
                                      100*self.proportion_routes_cancelled,
                                      100*self.proportion_gcc_size)))   
            plt.suptitle(title, fontsize=16)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.90,
                            bottom=0.05, wspace=0.15, hspace=0.05)

        ax = plt.subplot(111)

        # Draw the background of the map

        # Show Iceland or not in the map 
        # (y1, x1) are the coordinates of the lower left corner of the map
        # (y2, x2) are the coordinates of the upper right corner of the map
        if iceland:
            y1 = 34
            x1 = -25
            y2 = 70
            x2 = 67.7604
        else:
            y1 = 35
            x1 = -12
            y2 = 60
            x2 = 33
        base_map = Basemap(resolution="i", projection="merc", llcrnrlat=y1,
                    urcrnrlat=y2, llcrnrlon=x1, urcrnrlon=x2, lat_ts=(x1+x2)/2)
        base_map.drawcountries(linewidth=0.5)
        base_map.drawcoastlines(linewidth=0.5)
        base_map.fillcontinents(color="lightgrey", lake_color="white")
        base_map.drawmapboundary(fill_color="white")

        # Plot the circle of the impact

        centerlon = self.long_center
        centerlat = self.lat_center
        r = self.dist_from_center
        circle(base_map, centerlon, centerlat, r, lw=2., 
               label="Limit of hazard impact", color="red")

        # Plot the center of the impact
        base_map.scatter(centerlon, centerlat, latlon=True, alpha=0.8, 
                         marker="^", c="black", s=12**2, label="Hazard center", 
                         zorder=10)

        # Show the closed airports (scatter)
        closed_airports = self.airports_to_close
        lat_closed = [self.airports_info.loc[closed_airport]["latitude"] 
                      for closed_airport in closed_airports]
        long_closed = [self.airports_info.loc[closed_airport]["longitude"] 
                      for closed_airport in closed_airports]

        base_map.scatter(long_closed, lat_closed, latlon=True, alpha=0.5, c="r", 
                         s=10**2, label="Closed airports", zorder=10)

        # Plot the routes that have not been closed

        i = 0
        self.new_routes_permuted = self.new_routes.reindex(np.random.permutation(
                                                           self.new_routes.index))
        for index, row in self.new_routes_permuted.iterrows():
            lat = [self.airports_info.loc[row["source_airport"]]["latitude"], 
                   self.airports_info.loc[row["destination_airport"]]["latitude"]]
            lon = [self.airports_info.loc[row["source_airport"]]["longitude"], 
                   self.airports_info.loc[row["destination_airport"]]["longitude"]]
            x, y = base_map(lon, lat)
            base_map.plot(x, y,  linewidth=0.15, c=operational_routes_color, 
                          label="Operational routes" if i == 0 else "")
            i += 1
            if i == n:
                break

        # Plot the closed routes

        self.closed_routes_permuted = self.closed_routes.reindex(
            np.random.permutation(self.closed_routes.index))
        i = 0
        for index, row in self.closed_routes_permuted.iterrows():
            lat = [self.airports_info.loc[row["source_airport"]]["latitude"], 
                   self.airports_info.loc[row["destination_airport"]]["latitude"]]
            lon = [self.airports_info.loc[row["source_airport"]]["longitude"], 
                   self.airports_info.loc[row["destination_airport"]]["longitude"]]
            x, y = base_map(lon, lat)
            base_map.plot(x, y,  linewidth=0.15, c=closed_routes_color, 
                          label="Closed routes" if i == 0 else "")
            i += 1
            if i == n:
                break

        plt.legend(bbox_to_anchor=(0, 1), loc="upper right", ncol=1)

        plt.setp(plt.gca().get_legend().get_texts(), fontsize="14")

        if save_fig is not None:
            savefig("../figures/simulation_{}.png".format(str(save_fig)))

        plt.show()
