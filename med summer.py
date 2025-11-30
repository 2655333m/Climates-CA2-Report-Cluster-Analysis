#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:32:44 2025

@author: andrewmilne
"""

import netCDF4 as nc  # extracting the data
from netCDF4 import num2date  # extract time varible for seasons
import numpy as np # mathematical operations on large arrays 
import matplotlib.pyplot as plt # plot dendrogram
from sklearn.preprocessing import StandardScaler # standardisation 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # hierarchical clustering 
import cartopy.crs as ccrs # mapping the clusters
import cartopy.feature as cfeature  # mapping the clusters
from sklearn.cluster import KMeans # calcualte k-means

# Set working directory 
cwd = '/Users/andrewmilne/Desktop/GeoScience/Masters/Climatology/Assesments/CA2/Data'

# Load datasets
data_v10 = nc.Dataset('/Users/andrewmilne/Desktop/GeoScience/Masters/Climatology/Assesments/CA2/Data/v10.nc', 'r')
data_u10 = nc.Dataset('/Users/andrewmilne/Desktop/GeoScience/Masters/Climatology/Assesments/CA2/Data/u10.nc', 'r')
data_t2m = nc.Dataset('/Users/andrewmilne/Desktop/GeoScience/Masters/Climatology/Assesments/CA2/Data/t2m.nc', 'r')

# Extract lat/lon/time 
lat = data_v10.variables['latitude'][:]  
lon = data_v10.variables['longitude'][:]  
time = data_v10.variables['time'][:]

# Create subset for Mediterranean  subset
MED = (lat >= 30) & (lat <= 45)
MED_lon = (lon >= -10) & (lon <= 40)

# dictionary to store NH subsets
extratropics = {}

files = {
    "v10": data_v10,
    "u10": data_u10,
    "t2m": data_t2m
}

# Create loop of Mediterranean  to extraxt varibles
for varname, ds in files.items():
    field = ds.variables[varname][:]    
    extratropics[varname] = field[:, MED, :][:, :, MED_lon]
    
# Rearrange data arrays
time = num2date(ds['time'][:], ds['time'].units)

# Extract years and months via time step
years = []
months = []
for t in time:
    years.append(t.year)
    months.append(t.month)

# Create emppty mean dictionary 
mean = {}

# Convert monthly data --> summer means (june, july, august)
for varname, data in extratropics.items():
    n_time = data.shape[0]
    summer_months = [i for i, month in enumerate(months) if month in [6, 7, 8]] # Empty subset of summer, dec, jan, feb
    summer_data = data[summer_months]
    n_summers = len(summer_data) // 3
    reshape = summer_data[:n_summers*3].reshape(n_summers, 3, data.shape[1], data.shape[2])
    mean[varname] = np.mean(reshape, axis=1)

# Create data matrix
n_years = mean['t2m'].shape[0]
n_lat = mean['t2m'].shape[1] 
n_lon = mean['t2m'].shape[2]

matrix = np.zeros((n_lat * n_lon, n_years * 3))

# Convert 2d spatial gris to 1d list for matrix
for i in range(n_lat): # loop throughn each lattitude
    for j in range(n_lon): # loop throughn each longitude
        idx = i * n_lon + j
        t2m_ts = mean['t2m'][:, i, j]
        u10_ts = mean['u10'][:, i, j]
        v10_ts = mean['v10'][:, i, j]
        matrix[idx, :] = np.concatenate([t2m_ts, u10_ts, v10_ts])

# Remove any missing data for clustering 
missing_data = ~np.any(np.isnan(matrix), axis = 1)
clean = matrix[missing_data, :]

# Standardization, since varibles have different units and scales (consistency)
scale = StandardScaler()
scaled_data = scale.fit_transform(clean)
print(scaled_data)

Z = linkage(scaled_data, method='ward')

# Plot dendrogram 
# Adapted from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode='level', p=5)
plt.title(" Mediterranean – Hierarchical Clustering Dendrogram (summer Months)")
plt.xlabel("Cluster") 
plt.ylabel("Distance")
plt.show()

# clusters 
n_clusters = 4
labels = fcluster(Z, n_clusters, criterion='maxclust')

"""
Via the dendragram, we can identify 4 clusters.
We now move onto plotting these cluster in a cluster map
of the Mediterranean sea
"""

# Create an empty grid for clusters (lat x lon)
cluster_map = np.full((n_lat, n_lon), np.nan)

# Fill grid ONLY where data was not missing
clean_idx = np.where(missing_data)[0]

for k, idx in enumerate(clean_idx):
    i = idx // n_lon
    j = idx % n_lon
    cluster_map[i, j] = labels[k]

# Plot Cluster Map
# Artifical Intellegence assisted on plotting the map

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree()) # Cartopys Carrée projection

# Add outlines of continents onto figure
ax.coastlines(resolution='110m', linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Crop to focus on the Med 
ax.set_extent([-10, 40, 30, 45], crs=ccrs.PlateCarree())

# Plot the cluster map
plt.pcolormesh(
    lon[MED_lon], lat[MED], cluster_map,
    transform=ccrs.PlateCarree(),
    shading='nearest'
)

plt.title("Cluster Map – Mediterranean (summer Means)")
plt.colorbar(label="Cluster")
plt.show()

# K-Mean analysis
# Number of clusters
k = 4

# Initialise K-means
kmeans = KMeans(
    n_clusters=k,
    init='k-means++',
    max_iter=500,
    n_init=20,
    random_state=40
)

# Fit to your standardised data
kmeans.fit(scaled_data)

# Cluster labels for each grid pointwhat 
kmeans_labels = kmeans.labels_

# Centroids (cluster centres)
centroids = kmeans.cluster_centers_

print("K-means labels:", kmeans_labels)
print("Centroid matrix shape:", centroids.shape)

# Extract Varibles from Cluster as mEAN

# Calculate summer mean

t2m_summer_mean = np.mean(mean['t2m'], axis=0)   
u10_summer_mean = np.mean(mean['u10'], axis=0)
v10_summer_mean = np.mean(mean['v10'], axis=0)

# Flatten the 2D spatial grids to 1D
t2m_flat = t2m_summer_mean.reshape(-1)   
u10_flat = u10_summer_mean.reshape(-1)
v10_flat = v10_summer_mean.reshape(-1)

# Remove any missing points

t2m_flat_clean = t2m_flat[missing_data]
u10_flat_clean = u10_flat[missing_data]
v10_flat_clean = v10_flat[missing_data]

# Create cluster list
clusters = {}

for c in range(k):  # k = 4
    idx = np.where(kmeans_labels == c)[0]   # indices in the CLEAN array
    clusters[c] = {
        "t2m": t2m_flat_clean[idx],
        "u10": u10_flat_clean[idx],
        "v10": v10_flat_clean[idx]
    }

# Cluster mean 
for c in range(k):
    print(f"\nCluster {c}:")
    print("Mean t2m:", np.mean(clusters[c]["t2m"]))
    print("Mean u10:", np.mean(clusters[c]["u10"]))
    print("Mean v10:", np.mean(clusters[c]["v10"]))
