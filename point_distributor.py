import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon, MultiPoint

class PointsDistributor():
    """
    Evenly spacing of a certain number of points within an irregular polygon.
    In order to do so it first generates a grid of points 
    (defaults to 100x100, but can be modified according to the desired scale) 
    on top of any irregular polygon.
    
    Then it determines which points are within the polygon or outside of it.
    
    Finally it performs a k-means clustering of the inner points. 
    The k-means algorithm itself has inside its optimization objective function, 
    the goal of finding the maximum distance between clusters' centroids. 
    Therefore, it will automatically distribute a certain number of points 
    (the clusters' centroids) evenly within the polygon.
    
    """
    def __init__(self, coords, grid_size =(100,100)):
        """
        Input arguments:
            coords: polygon boundary coordinates (long,lat)
            grid_size: number of points in each side of the initial grid
        """
        self.coords = coords
        self.polygon = Polygon(self.coords) 
        self._generate_grid(grid_size)
    
    ### Auxiliary functions ###
    def _generate_grid(self, grid_size):
        self.dim_range = np.ptp(self.coords, axis = 0)
        delta = self.dim_range*[1/item for item in (100,100)]
        SW = (self.polygon.bounds[0], self.polygon.bounds[1])

        self.inner_pts = []
        self.outer_pts = []

        for i in range(100):
            for j in range(100):
                p = Point(SW + (i,j)*delta)
                if self.polygon.contains(p):
                    self.inner_pts.append(p)
                else:
                    self.outer_pts.append(p)
        self.X = np.array([pt.coords[0] for pt in self.inner_pts])
    
    def _create_inner_clusters(self):
        self.inner_clusters = []
        for label in range(self.n):
            my_members = (self.labels == label)
            pts = self.X[my_members, :]
            self.inner_clusters.append(MultiPoint(pts).convex_hull)
            
    ### Callable functions ###
    def view_polygon(self):
        centroid = self.polygon.centroid
        x_cen,y_cen = centroid.coords[0]
        x_pol,y_pol = self.polygon.exterior.xy

        fig, ax = plt.subplots()
        ax.plot(x_pol,y_pol)
        ax.plot(x_cen,y_cen, marker='o')
        plt.axis('off')
        plt.show()
        
    def view_grid(self):
        centroid = self.polygon.centroid
        x_cen,y_cen = centroid.coords[0]
        x_pol,y_pol = self.polygon.exterior.xy
        x_inner_pts = [pt.coords[0][0] for pt in self.inner_pts]
        y_inner_pts = [pt.coords[0][1] for pt in self.inner_pts]
        x_outer_pts = [pt.coords[0][0] for pt in self.outer_pts]
        y_outer_pts = [pt.coords[0][1] for pt in self.outer_pts]

        fig, ax = plt.subplots()
        ax.plot(x_pol,y_pol)
        ax.scatter(x_cen,y_cen, marker='o', s= 30, color= 'k')
        ax.scatter(x_inner_pts,y_inner_pts, marker='o', s= 0.5, color= 'b')
        ax.scatter(x_outer_pts,y_outer_pts, marker='o', s= 0.5, color= 'r')
        plt.axis('off')
        plt.show()
    
    def place_points(self, n):
        self.n = n
        k_means = KMeans(init="k-means++", n_clusters=n, n_init=10)
        k_means.fit(self.X)
        self.labels = k_means.labels_
        self.cluster_centers = k_means.cluster_centers_
        self._create_inner_clusters()
    
    def view_point_placement(self):
        fig,ax = plt.subplots()
        colors = plt.cm.Spectral(np.linspace(0, 1, len(set(self.labels))))
        for k, col in zip(range(len(self.cluster_centers)), colors):
            # create a list of all datapoints, where the datapoitns that are 
            # in the cluster (ex. cluster 0) are labeled as true, else they are
            # labeled as false.
            my_members = (self.labels == k)
            # define the centroid, or cluster center.
            cluster_center = self.cluster_centers[k]
            # plot the datapoints with color col.
            ax.plot(self.X[my_members, 0], self.X[my_members, 1], 'w', markerfacecolor=col, marker='.')
            # plot the centroids with specified color, but with a darker outline
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
        
        x_pol,y_pol = self.polygon.exterior.xy
        ax.plot(x_pol,y_pol)
        ax.set_title('{} uniformly distributed points'.format(self.n), fontsize = 12)
        plt.axis('off')
        plt.show()
    
    ### Results evaluation functions ###
    
    def results_summary(self):
        print('Number of points: {}'.format(self.n))
        print('Total area of primitive polygon: {}'.format(self.polygon.area))
        cluster_areas = [pol.area for pol in self.inner_clusters]
        cluster_area_mean = np.mean(cluster_areas)
        cluster_area_std_dev = np.std(cluster_areas)
        print('Inner cluster area distribution: {} ± {}'.format(cluster_area_mean,
                                                                cluster_area_std_dev
                                                               ))
    
    ### Data retrieving functions ###
    
    def get_inner_points(self):  
        return self.X
    
    def get_cluster_centers(self):
        return self.cluster_centers
    
    def get_labels(self):
        return self.labels
    
    def get_clusters(self):
        return self.inner_clusters