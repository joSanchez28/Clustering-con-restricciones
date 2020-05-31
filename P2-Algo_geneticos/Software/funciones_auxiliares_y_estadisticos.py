# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:41:45 2020

@author: jorge
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

# Funciones auxiliares:
def distance(x, y):
    return np.linalg.norm(x-y)

def V(x_i, x_j, partition, constraint_value): #constraint_value = -1,0,1
    """V(x_i, x_j) es la función que, dada una pareja de instancias, devuelve 1 si la asignación de
    dichas instancias viola una restricción y 0 en otro caso."""
    if (partition[x_i] != -1 and partition[x_j] != -1):
        if constraint_value == 1: #ML
            if partition[x_i] != partition[x_j]:
                return 1
        elif constraint_value == -1: #CL
            if partition[x_i] == partition[x_j]:
                return 1
    
    return 0

# ESTADÍSTICOS:
def mean_dist_intra_cluster(cluster_id, X,  partition, centroid = None):
    if centroid == None:
        centroid = np.mean(X[np.where(partition == cluster_id)[0]], axis = 0)
    return np.mean((X[np.where(partition == cluster_id)[0]] - centroid)**2, axis = 0) 

def general_deviation(X, partition, centroids = None):
    cluster_ids = np.unique(partition)
    if centroids == None:
        intra_cluster_mean_distances = [mean_dist_intra_cluster(cluster_id, X, partition, centroid = None) \
                                        for cluster_id in cluster_ids]
    else:
        intra_cluster_mean_distances = [mean_dist_intra_cluster(cluster_id, X, partition, centroid = centroids[cluster_id]) \
                                        for cluster_id in cluster_ids]
    return np.mean(intra_cluster_mean_distances)

def infeasibility(partition, const_list):
    return np.sum([V(x_i, x_j, partition, constraint_value) for (x_i,x_j,constraint_value) in const_list])

#Función objetivo
def max_dist(X):
    """Calcula la distancia máxima que hay entre dos instancias del conjunto de datos X"""
    max_distance = 0
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            dist = distance(X[i], X[j])
            if dist > max_distance:
                max_distance = dist
    return max_distance

def objective_func(X, partition, const_list, centroids = None, lambda_ = None):
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    return general_deviation(X, partition, centroids = centroids) + lambda_ * infeasibility(partition, const_list)

# Funciones de visualización: 
def visualise_iris_clusters(X, partition_sol, centroids = np.array([])):
    if centroids.size == 0:
        k = len(np.unique(partition_sol))
        centroids = np.zeros((k,X.shape[1]))
        for cluster_id in range(k):
            centroids[cluster_id] =  np.mean(X[np.where(partition_sol == cluster_id)[0]], axis = 0)
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))
    fig.suptitle('Representamos las instancias y los clusters según las diferentes características (2D)')
    axs[0, 0].scatter(X[:, 0], X[:, 1], s=40, c = partition_sol, cmap='viridis')
    axs[0, 0].scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5);
    axs[0, 0].set(xlabel='Característica 0', ylabel='Característica 1')
    axs[0, 1].scatter(X[:, 0], X[:, 2], s=40, c = partition_sol, cmap='viridis')
    axs[0, 1].scatter(centroids[:, 0], centroids[:, 2], c='black', s=200, alpha=0.5);
    axs[0, 1].set(xlabel='Característica 0', ylabel='Característica 2')
    axs[0, 2].scatter(X[:, 0], X[:, 3], s=40, c = partition_sol, cmap='viridis')
    axs[0, 2].scatter(centroids[:, 0], centroids[:, 3], c='black', s=200, alpha=0.5);
    axs[0, 2].set(xlabel='Característica 0', ylabel='Característica 3')

    axs[1, 0].scatter(X[:, 1], X[:, 0], s=40, c = partition_sol, cmap='viridis')
    axs[1, 0].scatter(centroids[:, 1], centroids[:, 0], c='black', s=200, alpha=0.5);
    axs[1, 0].set(xlabel='Característica 1', ylabel='Característica 0')
    axs[1, 1].scatter(X[:, 1], X[:, 2], s=40, c = partition_sol, cmap='viridis')
    axs[1, 1].scatter(centroids[:, 1], centroids[:, 2], c='black', s=200, alpha=0.5);
    axs[1, 1].set(xlabel='Característica 1', ylabel='Característica 2')
    axs[1, 2].scatter(X[:, 1], X[:, 3], s=40, c = partition_sol, cmap='viridis')
    axs[1, 2].scatter(centroids[:, 1], centroids[:, 3], c='black', s=200, alpha=0.5);
    axs[1, 2].set(xlabel='Característica 1', ylabel='Característica 3')

    axs[2, 0].scatter(X[:, 2], X[:, 0], s=40, c = partition_sol, cmap='viridis')
    axs[2, 0].scatter(centroids[:, 2], centroids[:, 0], c='black', s=200, alpha=0.5);
    axs[2, 0].set(xlabel='Característica 2', ylabel='Característica 0')
    axs[2, 1].scatter(X[:, 2], X[:, 1], s=40, c = partition_sol, cmap='viridis')
    axs[2, 1].scatter(centroids[:, 2], centroids[:, 1], c='black', s=200, alpha=0.5);
    axs[2, 1].set(xlabel='Característica 2', ylabel='Característica 1')
    axs[2, 2].scatter(X[:, 2], X[:, 3], s=40, c = partition_sol, cmap='viridis')
    axs[2, 2].scatter(centroids[:, 2], centroids[:, 3], c='black', s=200, alpha=0.5);
    axs[2, 2].set(xlabel='Característica 2', ylabel='Característica 3')

    axs[3, 0].scatter(X[:, 3], X[:, 0], s=40, c = partition_sol, cmap='viridis')
    axs[3, 0].scatter(centroids[:, 3], centroids[:, 0], c='black', s=200, alpha=0.5);
    axs[3, 0].set(xlabel='Característica 3', ylabel='Característica 0')
    axs[3, 1].scatter(X[:, 3], X[:, 1], s=40, c = partition_sol, cmap='viridis')
    axs[3, 1].scatter(centroids[:, 3], centroids[:, 1], c='black', s=200, alpha=0.5);
    axs[3, 1].set(xlabel='Característica 3', ylabel='Característica 1')
    axs[3, 2].scatter(X[:, 3], X[:, 2], s=40, c = partition_sol, cmap='viridis')
    axs[3, 2].scatter(centroids[:, 3], centroids[:, 2], c='black', s=200, alpha=0.5);
    axs[3, 2].set(xlabel='Característica 3', ylabel='Característica 2')
    
def visualise_rand_clusters(X, partition_sol, centroids = np.array([])):
    if centroids.size == 0:
        k = len(np.unique(partition_sol))
        centroids = np.zeros((k,X.shape[1]))
        for cluster_id in range(k):
            centroids[cluster_id] =  np.mean(X[np.where(partition_sol == cluster_id)[0]], axis = 0)
    plt.scatter(X[:, 0], X[:, 1], s=40, c = partition_sol, cmap='viridis')
    plt.scatter(centroids[:,0], centroids[:,1], c='black', s=200, alpha=0.5);
    plt.xlabel('Característica 0')
    plt.ylabel('Característica 1')
    