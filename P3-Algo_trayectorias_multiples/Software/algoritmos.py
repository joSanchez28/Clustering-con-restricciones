# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:37:31 2020

@author: jorge
"""
import numpy as np
import random
import copy
from funciones_auxiliares_y_estadisticos import *

#---------------------------------ALGORITMOS PRÁCTICA 1------------------------
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

def generate_valid_initial_sol(X, k):
    valid_partition = False
    while not valid_partition:
        partition = generate_initial_sol(X, k)
        unique, counts = np.unique(partition, return_counts=True)
        assignations_counter = dict(zip(unique, counts))
        if len(unique)== k:
            valid_partition = True
    return partition, assignations_counter


### Algoritmo -
def local_search_from_partition(partition, assignations_counter, X, const_matrix, const_list, k, lambda_ = None, 
                                max_evaluations = 100000, restart_mode = False):
    n_instances = X.shape[0]
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    current_func_value = objective_func(X, partition, const_list, centroids = None, lambda_ = lambda_)
    counter = 1 #Contamos el número de veces que se evalua la función objetivo
    #Creamos una lista de parejas [(0,+1),(0,+2),..,(1,+1),(1,+2),...,(n_instances, +1),...] Esta lista representará todas las 
    #operaciones de movimiento posibles desde una particion dada. Esto será lo que barajemos.
    virtual_neighborhood = [(index, to_add) for index in range(n_instances) for to_add in range(1, k)]
    found_better_sol = True
    while (counter < max_evaluations and found_better_sol): #Condición-Parada: Encontrar mejor solución de un entorno o 100000 evaluaciones.
        found_better_sol = False
        np.random.shuffle(virtual_neighborhood)
        #for operation in virtual_neighborhood:
        i = 0
        while (counter < max_evaluations and (not found_better_sol) and i<n_instances):
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
    
    if restart_mode:
        return partition, current_func_value
    else:
        return partition    
    
    
def local_search(X, const_matrix, const_list, k, lambda_ = None, max_evaluations = 100000, restart_mode = False):
    #Declaramos un diccionario 'assignations_counter' para vigilar cuantas instancias tiene cada cluster 
    #asignadas (para asegurar que nunca tienen 0)
    partition, assignations_counter = generate_valid_initial_sol(X, k)
    return local_search_from_partition(partition, assignations_counter, X, const_matrix, const_list, k, 
                                       lambda_ = None, max_evaluations = max_evaluations, restart_mode = restart_mode)



#---------------------------------ALGORITMOS PRÁCTICA 3------------------------
##Enfriamiento simulado
def initial_temperature(mu, initial_cost, phi):
    return ((mu*initial_cost)/-(np.log(phi)))

def cauchy_scheme(current_temp, beta):
    next_temp = current_temp/(1+beta*current_temp)
    return next_temp

def simulated_annealing_from_partition(current_partition, assignations_counter, X, const_matrix, const_list, k, lambda_ = None,
                                       max_evaluations = 100000, restart_mode = False):
    n_instances = X.shape[0]
    best_partition = current_partition
    current_func_value = objective_func(X, current_partition, const_list, centroids = None, lambda_ = lambda_)
    best_func_value = current_func_value
    #counter = 1 #Contamos las veces que se evalua la función objetivo #No es necesario gracias a m_iterations
    #Calculamos la temperatura inicial
    initial_temp = initial_temperature(0.3, best_func_value, 0.3)
    current_temp = initial_temp
    final_temp = 10**(-3)
    while final_temp >= initial_temp:
        final_temp *= 10**(-3)
    #Escalamos max_neighbours (que está a priori pensado para ejecuciones individuales de 100000 evaluaciones)
    max_neighbours = 10*n_instances * max_evaluations/100000 
    max_successes = 0.1*max_neighbours
    m_iterations = max_evaluations/max_neighbours
    #print("max_neighbours: ", max_neighbours)
    #print("max_successes: ", max_successes)
    #print("m_iterations: ", m_iterations)
    beta = (initial_temp - final_temp)/(m_iterations * initial_temp * final_temp)
    annealings = 0
    successes = 1
    while annealings<m_iterations and successes > 0:
        successes = 0
        generated_neighbours = 0
        while generated_neighbours < max_neighbours and successes < max_successes:
            #Generamos un vecino
            gen_idx = random.randint(0,n_instances-1)
            #Comprobamos que el cambio en el gen elegido no rompe las restricciones (fuertes)
            while assignations_counter[current_partition[gen_idx]] <= 1:
                gen_idx = random.randint(0,n_instances-1)
            previous_gen_value = current_partition[gen_idx]
            current_partition[gen_idx] = (current_partition[gen_idx] + random.randint(1,k-1)) % k
            generated_neighbours += 1
            candidate_func_value = objective_func(X, current_partition, const_list, centroids = None, lambda_ = lambda_)
            #counter += 1
            increment_f = candidate_func_value - current_func_value 
            if (increment_f < 0 or random.random() <= np.exp(-increment_f/current_temp)):
                successes += 1 
                current_func_value = candidate_func_value
                assignations_counter[previous_gen_value] -= 1
                assignations_counter[current_partition[gen_idx]] += 1
                if current_func_value < best_func_value:
                    best_func_value = current_func_value
                    best_partition = current_partition
            else:
                current_partition[gen_idx] = previous_gen_value #Si no es aceptada volvemos a la partición anterior
        annealings += 1
        #print("Temperatura actual: ", current_temp)
        current_temp = cauchy_scheme(current_temp, beta)
        #print(counter, end="\r", flush=True)
    
    #print("Número de exitos en el último enfriamiento: ", successes)
    #print("Número total de evaluaciones realizadas de la función objetivo: ", counter)
    if restart_mode:
        return best_partition, best_func_value
    else:
        return best_partition 


def simulated_annealing_algo(X, const_matrix, const_list, k, lambda_ = None, max_evaluations = 100000, restart_mode = False):
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    #Generamos la solución inicial
    current_partition, assignations_counter = generate_valid_initial_sol(X, k)
    return simulated_annealing_from_partition(current_partition, assignations_counter, X, const_matrix, const_list, k, 
                                              lambda_ = lambda_, max_evaluations = max_evaluations, restart_mode = restart_mode)

## Búsqueda Multiarranque Básica BMB
def random_restart_local_search(X, const_matrix, const_list, k, lambda_ = None, n_ls = 10, evaluations_per_ls = 10000):
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    best_partition, best_func_value = local_search(X, const_matrix, const_list, k, lambda_ = lambda_, 
                                                   max_evaluations = evaluations_per_ls, restart_mode = True)
    for i in range(n_ls-1):
        partition, func_value = local_search(X, const_matrix, const_list, k, lambda_ = lambda_, 
                                             max_evaluations = evaluations_per_ls, restart_mode = True)
        if func_value < best_func_value:
            best_func_value = func_value
            best_partition = partition
    return best_partition

## Búsqueda local reiterada (ILS)
###Operador de Mutación por segmento
####Este operador no cambia el objeto "partition", sino que crea un nuevo objeto y lo devuelve
def segment_mutation_operator(partition, segment_length, k): 
    n = len(partition)
    segment_ini = random.randint(0, n-1)
    segment_indices = [i%n for i in range(segment_ini, segment_ini + segment_length)]
    mut_partition = [partition[i] for i in range(n)]
    for i in segment_indices:
        mut_partition[i] = random.randint(0, k-1)
    return mut_partition

#El operador de mutación por segmento podría dar lugar a soluciones que rompieran la restricción fuerte de
#que cada cluster tenga al menos una instancia asignada, por lo que necesitamos una función que las
#repare en dicho caso.
def repair_partition(partition, assignations_counter, k):
    for i in range(k):
        if i not in assignations_counter: #El i-ésimo cluster no está en la partición
            gen_idx = random.randint(0,len(partition)-1)
            while assignations_counter[partition[gen_idx]] <= 1: #Nos aseguramos de que la reparación no rompe otra restricción
                gen_idx = random.randint(0,len(partition)-1)
            assignations_counter[partition[gen_idx]] -= 1
            partition[gen_idx] = i
            assignations_counter[i] = 1
            
def iterated_local_search(X, const_matrix, const_list, k, lambda_ = None, n_ls = 10, evaluations_per_ls = 10000, 
                         search_algo = local_search_from_partition):
    n_instances = X.shape[0]
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    best_partition, assignations_counter = generate_valid_initial_sol(X, k)
    best_partition, best_func_value = search_algo(best_partition, assignations_counter, X, const_matrix, const_list, k, 
                                                     lambda_ = lambda_, max_evaluations = evaluations_per_ls, 
                                                     restart_mode = True)
    for i in range(n_ls-1):
        candidate_partition = segment_mutation_operator(best_partition, segment_length = int(n_instances * 0.1), k=k)
        unique, counts = np.unique(candidate_partition, return_counts=True)
        assignations_counter = dict(zip(unique, counts))
        if len(unique) != k:
            repair_partition(candidate_partition, assignations_counter, k)
        candidate_partition, func_value = search_algo(candidate_partition, assignations_counter, X, const_matrix, 
                                                         const_list, k, lambda_ = lambda_, 
                                                         max_evaluations = evaluations_per_ls, restart_mode = True)
        if func_value < best_func_value: #Criterio de aceptación el mejor.
            best_func_value = func_value
            best_partition = candidate_partition
        
    return best_partition


def hybrid_iterated_local_search(X, const_matrix, const_list, k, lambda_ = None, n_ls = 10, 
                                 evaluations_per_ls = 10000):
    return iterated_local_search(X, const_matrix, const_list, k, lambda_ = lambda_, n_ls = n_ls, 
                                 evaluations_per_ls = evaluations_per_ls, 
                                 search_algo = simulated_annealing_from_partition)

