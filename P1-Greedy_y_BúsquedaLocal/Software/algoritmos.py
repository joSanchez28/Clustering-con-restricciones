# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:37:31 2020

@author: jorge
"""
import numpy as np
import copy
from funciones_auxiliares_y_estadisticos import *

#ALGORITMO COPKM
### Funciones previas -
def increment_in_inf(x_i, cluster_id, partition, const_matrix):
    """Calcula el incremento en infeasibility provocado por asignar la instancia x_i al cluster indicado"""
    increment = 0
    partition[x_i] = cluster_id #Temporalmente cambiamos la partición para ver el incremento en infeasibility
    for x_j in np.where(partition != -1)[0] :
        increment += V(x_i, x_j, partition, const_matrix[x_i,x_j])
    partition[x_i] = -1 #Volvemos a la partición anterior (pues aun está por decidir a qué cluster se asocia x_i)
    return increment

def centroids_initialization(X, k):
    #Calculamos los dominios de cada característica (columna) del conjunto de datos
    domines = np.array([np.min(X, axis = 0), np.max(X, axis = 0)]).transpose()
    centroids = []
    for j in range(k):
        centroid_j = [np.random.uniform(low=domines[i][0], high=domines[i][1]) for i in range(domines.shape[0])]
        centroids.append(centroid_j)
    return centroids

### Algoritmo -
def copkm_algorithm(X, const_matrix, const_list, initial_centroids):
    k = len(initial_centroids)
    #Barajamos los índices para recorrer X de forma aleatoria sin repetición.
    rsi = np.arange(X.shape[0])
    np.random.shuffle(rsi)
    #Inicializamos la partición; la representamos como un vector en el que el valor en la posición i indica el cluster
    #asignado a la instancia x_i (como al principio no tenemos ninguna asignación, ponemos todos los valores a -1)
    partition = np.full(X.shape[0], -1)
    prev_partition = np.full(X.shape[0], -1)
    centroids = copy.deepcopy(initial_centroids)
    change = True
    cycle = False
    #iterations = 0
    while (change and not cycle):
        #print("Iteration ", iterations)
        change = False
        new_partition = np.full(X.shape[0], -1)
        #Asignamos las instancias a cada cluster
        for i in rsi:
            #Calculamos el incremento en infeasibility que produce la asignación de x_i a cada cluster c_j 
            #y guardamos las js que producen menos incremento
            increments = np.array([increment_in_inf(i, j, new_partition, const_matrix) for j in range(k)])
            less_incr_cluster = np.sort(np.where(increments == increments.min())[0])          
            #De entre las asignaciones (js) que producen menos incremento en infeasibility, seleccionar la asociada con el
            #centroide mu_j más cercano a x_i
            distances = [distance(X[i], centroids[j]) for j in less_incr_cluster]
            closest = less_incr_cluster[np.argmin(distances)]
            #Elegimos closest como el cluster para la instancia x_i
            new_partition[i] = closest
            if new_partition[i] != partition[i]:
                change = True
        if np.array_equal(prev_partition, new_partition):
            cycle = True
        prev_partition = partition
        partition = new_partition
        #Actualizamos los centroides de cada cluster
        for j in range(k):
            centroids[j] = np.mean(X[np.where(partition == j)[0]], axis = 0)
        #iterations +=1
    return partition

###Algoritmo copkm con la inicialización aleatoria de clusters incluida:
def copkm_algorithm_with_ini(X, const_matrix, const_list, k):
    initial_centroids = centroids_initialization(X, k)
    return copkm_algorithm(X, const_matrix, const_list, initial_centroids)
            
#ALGORITMO DE BÚSQUEDA LOCAL:
### Funciones previas -
def generate_initial_sol(X, k):
    return np.random.randint(low=0, high=k, size=X.shape[0])

def check_validity(assignations_counter):
    return (min(assignations_counter.values()) > 0)

### Algoritmo -
def local_search(X, const_matrix, const_list, k, lambda_ = None):
    n_instances = X.shape[0]
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    #Declaramos un diccionario 'assignations_counter' para vigilar cuantas instancias tiene cada cluster 
    #asignadas (para asegurar que nunca tienen 0)
    valid_partition = False
    while not valid_partition:
        partition = generate_initial_sol(X, k)
        unique, counts = np.unique(partition, return_counts=True)
        assignations_counter = dict(zip(unique, counts))
        if check_validity(assignations_counter):
            valid_partition = True
    current_func_value = objective_func(X, partition, const_list, centroids = None, lambda_ = lambda_)
    counter = 1 #Contamos el número de veces que se evalua la función objetivo
    #Creamos una lista de parejas [(0,+1),(0,+2),..,(1,+1),(1,+2),...,(n_instances, +1),...] Esta lista representará todas las 
    #operaciones de movimiento posibles desde una particion dada. Esto será lo que barajemos.
    virtual_neighborhood = [(index, to_add) for index in range(n_instances) for to_add in range(1, k)]
    found_better_sol = True
    while (counter <= 100000 and found_better_sol): #Condición-Parada: Encontrar mejor solución de un entorno o 100000 evaluaciones.
        found_better_sol = False
        np.random.shuffle(virtual_neighborhood)
        #for operation in virtual_neighborhood:
        i = 0
        while (counter <= 100000 and (not found_better_sol) and i<n_instances):
            operation = virtual_neighborhood[i]
            tmp = partition[operation[0]]
            #Ejecutamos la operación
            partition[operation[0]] = (partition[operation[0]] + operation[1]) % k
            func_val = objective_func(X, partition, const_list, centroids = None, lambda_ = lambda_)
            counter += 1
            #Si la operación nos lleva a una mejor partición que sea válida, nos quedamos con ella.
            if (func_val < current_func_value and assignations_counter[tmp] > 1): 
                assignations_counter[tmp] -= 1
                assignations_counter[partition[operation[0]]] += 1
                current_func_value = func_val
                found_better_sol = True
            else: #Si no, volvemos a la particición anterior y probamos con la siguiente operación
                partition[operation[0]] = tmp 
            i += 1
    return partition    