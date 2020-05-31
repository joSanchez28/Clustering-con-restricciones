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

def check_validity(assignations_counter, k):
    return (len(assignations_counter.keys()) == k and min(assignations_counter.values()) > 0)

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
        if check_validity(assignations_counter, k):
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


#---------------------------------ALGORITMOS PRÁCTICA 2------------------------
##Operadores de cruce:
###Operador de cruce uniforme
def uniform_cross_operator(father1, father2):
    n = len(father1)
    shuffle_indices = list(range(n))
    random.shuffle(shuffle_indices)
    child = [0]*n
    for i in range(int(n/2)): #Si n es impar, se asigna uno más del padre 2
        child[shuffle_indices[i]] = father1[shuffle_indices[i]]
    for i in range(int(n/2), n):
        child[shuffle_indices[i]] = father2[shuffle_indices[i]]
    return child

###Operador de cruce por segmento fijo
def fixed_segment_operator(father1, father2):
    n = len(father1)
    segment_ini = random.randint(0, n-1)
    segment_lenght = random.randint(0, n-1)
    segment_indices = [i%n for i in range(segment_ini, segment_ini + segment_lenght)]
    child = [-1]*n
    for i in segment_indices:
        child[i] = father1[i]
    #Para el resto de genes que no están en el segmento: Cruce uniforme
    indices = [i for i, x in enumerate(child) if x == -1]
    random.shuffle(indices)
    for i in range(int(len(indices)/2)): #Si len(indices) es impar, se asigna uno más del padre 2
        child[indices[i]] = father1[indices[i]]
    for i in range(int(len(indices)/2), len(indices)):
        child[indices[i]] = father2[indices[i]]
    return child

#Ambos operadores de cruce podrían dar lugar a soluciones que rompieran la restricción fuerte de
#que cada cluster tenga al menos una instancia asignada, por lo que necesitamos una función que las
#repare en dicho caso.
def repair_partition(partition, assignations_counter, k):
    for i in range(k):
        if i not in assignations_counter: #Si el i-ésimo cluster no está en la partición
            gen_idx = random.randint(0,len(partition)-1)
            while assignations_counter[partition[gen_idx]] <= 1: #Nos aseguramos de que la reparación no rompe otra restricción
                gen_idx = random.randint(0,len(partition)-1)
            assignations_counter[partition[gen_idx]] -= 1
            partition[gen_idx] = i
            assignations_counter[i] = 1
            
#ALGORITMOS GENÉTICOS
# Algoritmo genético generacional (AGG) con elitismo 
def generate_initial_population(X, const_list, k, lambda_, population_size, counter):
    current_population = []
    for i in range(population_size):
        #Declaramos diccionarios 'assignations_counter' para vigilar cuantas instancias tiene cada cluster 
        #asignadas (para asegurar que nunca tienen 0) en cada vector solución (o cromosoma)
        valid_partition = False
        while not valid_partition:
            partition = generate_initial_sol(X, k)
            unique, counts = np.unique(partition, return_counts=True)
            assignations_counter = dict(zip(unique, counts))
            if len(unique) == k: 
                valid_partition = True
        func_value = objective_func(X, partition, const_list, centroids = None, lambda_ = lambda_)
        counter += 1
        current_population.append([partition, func_value, assignations_counter])
    return current_population, counter

def new_generation(X, const_list, k, lambda_, population_size, counter, n_cross_expected, n_instances,
                   cross_operator, n_mutations_expected, current_population):
    new_population = []
    for i in range(population_size):
        #Torneo binario - Aprovechamos que la población está ordenada 
        father_idx = min(random.sample(range(population_size), 2)) #Saca dos elementos SIN repetición y se queda con el min.
        new_population.append(current_population[father_idx]) 

    #Aplicamos el operador de cruce a las parejas de forma ordenada
    for i in range(n_cross_expected):
        father1_idx = i*2
        father1 = new_population[father1_idx][0]
        father2 = new_population[father1_idx + 1][0]
        for j in range(2):
            child_partition = cross_operator(father1, father2)
            unique, counts = np.unique(child_partition, return_counts=True) 
            assignations_counter = dict(zip(unique, counts))
            if len(unique) < k: 
                repair_partition(child_partition, assignations_counter, k)
            #No llamamos aún a la función objetivo porque sería un malgasto si estas nuevas soluciones mutaran...
            new_population[father1_idx + j] = [child_partition, -1, assignations_counter]
    #Mutaciones:
    for i in range(n_mutations_expected):
        cromo_idx = random.randint(0,population_size-1)
        gen_idx = random.randint(0,n_instances-1)
        #Comprobamos que la mutación en el gen elegido no rompe las restricciones (fuertes)
        while new_population[cromo_idx][2][new_population[cromo_idx][0][gen_idx]] <= 1:
            gen_idx = random.randint(0,n_instances-1)
        new_population[cromo_idx][2][new_population[cromo_idx][0][gen_idx]] -= 1
        new_population[cromo_idx][0][gen_idx] = (new_population[cromo_idx][0][gen_idx] + random.randint(1,k-1)) % k
        new_population[cromo_idx][2][new_population[cromo_idx][0][gen_idx]] += 1
        new_population[cromo_idx][1] = -1
    #Llamamos a la función objetivo el número de veces estrictamente necesario (solo para los cromosomas nuevos)
    for cromo in new_population:
        if cromo[1] == -1:
            cromo[1] = objective_func(X, cromo[0], const_list, centroids = None, lambda_ = lambda_)
            counter += 1
    return new_population, counter

def generational_genetic_algo(X, const_matrix, const_list, k, lambda_ = None, cross_operator = uniform_cross_operator,
                              population_size = 50, cross_prob = 0.7, mutation_prob = 0.001):
    n_instances = X.shape[0]
    n_cross_expected = int(cross_prob * population_size / 2) #Número esperado de cruces (2 hijos por cruce)
    n_mutations_expected = int(mutation_prob * population_size * n_instances) #Número esperado de mutaciones
    #print("Número esperado de cruces (2 hijos por cruce): ", n_cross_expected)
    #print("Número esperado de mutaciones: ", n_mutations_expected)
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    counter = 0 #Contamos el número de veces que se evalua la función objetivo
    #Generamos la población inicial:
    current_population, counter = generate_initial_population(X, const_list, k, lambda_, population_size, counter)
    current_population.sort(key=lambda x:x[1]) #Ordenamos la población según la calidad de los cromosomas (de mejor a peor)
    while (counter < 100000): 
        new_population, counter = new_generation(X, const_list, k, lambda_, population_size, counter, n_cross_expected, 
                                                 n_instances, cross_operator, n_mutations_expected, current_population)
        #Ordenamos la nueva población
        new_population.sort(key=lambda x:x[1]) #Ordenamos la población según la calidad de los cromosomas
        #Aplicamos el elitismo
        if current_population[0][1] < new_population[0][1]: 
            new_population[population_size-1] = current_population[0]
            new_population.sort(key=lambda x:x[1])
        current_population = new_population
        
        #print(counter, end="\r", flush=True)
    return current_population[0][0]                   

##ALgoritmo con el operador de cruce uniforme fijado
def generational_genetic_algo_UN(X, const_matrix, const_list, k, lambda_ = None, 
                                 population_size = 50, cross_prob = 0.7, mutation_prob = 0.001):
    return generational_genetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, 
                                        cross_operator = uniform_cross_operator,
                                        population_size = population_size, cross_prob = cross_prob, 
                                        mutation_prob = mutation_prob)
    
##Algoritmo con el operador de cruce por segmento fijo fijado
def generational_genetic_algo_SF(X, const_matrix, const_list, k, lambda_ = None,
                              population_size = 50, cross_prob = 0.7, mutation_prob = 0.001):
    return generational_genetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, 
                                        cross_operator = fixed_segment_operator,
                                        population_size = population_size, cross_prob = cross_prob, 
                                        mutation_prob = mutation_prob)
    
# Algoritmo genético estacionario (AGE)
def stable_genetic_algo(X, const_matrix, const_list, k, lambda_ = None, cross_operator = uniform_cross_operator,
                              population_size = 50, mutation_prob = 0.001): #cross_prob = 1,
    n_instances = X.shape[0]
    cromo_mutation_prob = mutation_prob * n_instances #Prob. de mutación por CROMOSOMA
    #print("Probabilidad de mutación por cromosoma: ", cromo_mutation_prob)
    #print("Operador de cruce: ", str(cross_operator))
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    counter = 0 #Contamos el número de veces que se evalua la función objetivo
    current_population, counter = generate_initial_population(X, const_list, k, lambda_, population_size, counter)
    current_population.sort(key=lambda x:x[1]) #Ordenamos la población según la calidad de los cromosomas (de mejor a peor)
    while (counter < 100000): 
        parents = []
        for i in range(2):
            #Torneo binario - Aprovechamos que la población está ordenada 
            father_idx = min(random.sample(range(population_size), 2)) #Saca dos elementos SIN repetición y se queda con el min.
            parents.append(current_population[father_idx]) 
        #Aplicamos el operador de cruce a la pareja de padres
        #(La probabilidad de cruce en el algo estacionario es de 1)
        children = []
        for i in range(2):
                #Cruce:
                child_partition = cross_operator(parents[0][0], parents[1][0])
                unique, counts = np.unique(child_partition, return_counts=True) 
                assignations_counter = dict(zip(unique, counts))
                if len(unique) < k:
                    repair_partition(child_partition, assignations_counter, k)
                #Mutación:
                if random.random() <= cromo_mutation_prob: 
                    gen_idx = random.randint(0,n_instances-1)
                    #Comprobamos que la mutación en el gen elegido no rompe las restricciones (fuertes)
                    while assignations_counter[child_partition[gen_idx]] <= 1:
                        gen_idx = random.randint(0,n_instances-1)
                    assignations_counter[child_partition[gen_idx]] -= 1
                    child_partition[gen_idx] = (child_partition[gen_idx] + random.randint(1,k-1)) % k
                    assignations_counter[child_partition[gen_idx]] += 1
                func_value = objective_func(X, child_partition, const_list, centroids = None, lambda_ = lambda_)
                counter += 1
                children.append([child_partition, func_value, assignations_counter])                    
        #Sustituimos los hijos generados tras el cruce y la mutación por los dos peores de la población actual si los mejoran
        children += [current_population[-2], current_population[-1]]
        children.sort(key=lambda x:x[1])
        current_population[-2] = children[0]
        current_population[-1] = children[1]
        current_population.sort(key=lambda x:x[1])
        
        #print(counter, end="\r", flush=True)
    return current_population[0][0]           

## Algoritmo genético estacionario con operador de cruce uniforme fijado
def stable_genetic_algo_UN(X, const_matrix, const_list, k, lambda_ = None, 
                              population_size = 50, mutation_prob = 0.001):
    return stable_genetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, 
                               cross_operator = uniform_cross_operator, 
                               population_size = population_size, mutation_prob = mutation_prob)
 
## Algoritmo genético estacionario con operador de cruce por segmento fijo fijado
def stable_genetic_algo_SF(X, const_matrix, const_list, k, lambda_ = None, 
                              population_size = 50, mutation_prob = 0.001):
    return stable_genetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, 
                               cross_operator = fixed_segment_operator, 
                               population_size = population_size, mutation_prob = mutation_prob)
    
# Algoritmo memético
##Búsqueda local suave
def smooth_local_search(X, const_list, partition, current_func_value, 
                        assignations_counter, k, max_failures, counter, lambda_):
    shuffle_indices = list(range(len(partition)))
    random.shuffle(shuffle_indices)
    failures = 0
    improvement = True
    best_func_value = current_func_value
    i = 0
    while (improvement or (failures < max_failures)) and i < len(partition):
        improvement = False
        #Asignar el mejor cluster posible al gen shuffle_indices[i]
        if assignations_counter[partition[shuffle_indices[i]]] > 1: #Si cambiar este gen no rompe una restricción fuerte...
            best_cluster = partition[shuffle_indices[i]]
            for j in range(k):
                if j != partition[shuffle_indices[i]]:
                    partition[shuffle_indices[i]] = j
                    func_value = objective_func(X, partition, const_list, centroids = None, lambda_ = lambda_)
                    counter += 1
                    if func_value < best_func_value:
                        assignations_counter[best_cluster] -=1
                        assignations_counter[j] +=1
                        best_func_value = func_value
                        best_cluster = j
                        improvement = True
                    else: #Si no es mejor, volvemos a la asignación anterior
                        partition[shuffle_indices[i]] = best_cluster
        if improvement == False:
            failures += 1
        i += 1
    return partition, best_func_value, assignations_counter, counter        

def memetic_algo(X, const_matrix, const_list, k, lambda_ = None, cross_operator = fixed_segment_operator, #(el mejor operador)
                 population_size = 10, cross_prob = 0.7, mutation_prob = 0.001,
                 generation_per_ls = 10, perc_ls = 1.0, best_population = False): 
    #generation_per_ls - generaciones antes de hacer una BLS, 
    #perc_ls - porcentaje de la población sobre el que hacer una BLS,
    #best_population - si coger los mejores en este porcentaje o no.
    n_instances = X.shape[0]
    n_cross_expected = int(cross_prob * population_size / 2) #Número esperado de cruces (2 hijos por cruce)
    n_mutations_expected = int(mutation_prob * population_size * n_instances) #Número esperado de mutaciones
    n_solutions_for_local_search = int(population_size * perc_ls)
    max_failures = int(0.1 * n_instances)
    #print("Número esperado de cruces (2 hijos por cruce): ", n_cross_expected)
    #print("Número esperado de mutaciones: ", n_mutations_expected)
    if lambda_ == None:
        lambda_ = max_dist(X) / len(const_list)
    counter = 0 #Contamos el número de veces que se evalua la función objetivo
    #Generamos la población inicial:
    current_population, counter = generate_initial_population(X, const_list, k, lambda_, population_size, counter)
    current_population.sort(key=lambda x:x[1]) #Ordenamos la población según la calidad de los cromosomas (de mejor a peor)
    while (counter < 100000): 
        for generation in range(generation_per_ls):
            new_population, counter = new_generation(X, const_list, k, lambda_, population_size, counter, n_cross_expected, 
                                                 n_instances, cross_operator, n_mutations_expected, current_population)
            #Ordenamos la nueva población
            new_population.sort(key=lambda x:x[1]) #Ordenamos la población según la calidad de los cromosomas (de mejor a peor)
            #Aplicamos el elitismo
            if current_population[0][1] < new_population[0][1]: 
                new_population[population_size-1] = current_population[0]
                new_population.sort(key=lambda x:x[1])
            current_population = new_population
        #Aplicamos la búsqueda local suave a la proporción de la población determinada por perc_ls y best_population.
        if best_population:
            indices_for_ls = list(range(n_solutions_for_local_search))
        else:
            indices_for_ls = random.sample(range(population_size), n_solutions_for_local_search)
        for i in indices_for_ls:
            partition, func_value, assignations_counter, counter = smooth_local_search(X, const_list,
                                                                                       current_population[i][0], 
                                                                                       current_population[i][1], 
                                                                                       current_population[i][2], k, 
                                                                                       max_failures , counter, lambda_)
            current_population[i] = [partition, func_value, assignations_counter]
        
        #print(counter, end="\r", flush=True)
    return current_population[0][0]
    
#### AM-(10,1.0)
def memetic_algo_v1(X, const_matrix, const_list, k, lambda_ = None, cross_operator = fixed_segment_operator,
                 population_size = 10, cross_prob = 0.7, mutation_prob = 0.001):
    return memetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, cross_operator = cross_operator, 
                 population_size = population_size, cross_prob = cross_prob, mutation_prob = mutation_prob,
                 generation_per_ls = 10, perc_ls = 1.0, best_population = False)

#### AM-(10,0.1)
def memetic_algo_v2(X, const_matrix, const_list, k, lambda_ = None, cross_operator = fixed_segment_operator, 
                 population_size = 10, cross_prob = 0.7, mutation_prob = 0.001):
    return memetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, cross_operator = cross_operator, 
                 population_size = population_size, cross_prob = cross_prob, mutation_prob = mutation_prob,
                 generation_per_ls = 10, perc_ls = 0.1, best_population = False)

#### AM-(10,0.1mej)
def memetic_algo_v3(X, const_matrix, const_list, k, lambda_ = None, cross_operator = fixed_segment_operator, 
                 population_size = 10, cross_prob = 0.7, mutation_prob = 0.001):
    return memetic_algo(X, const_matrix, const_list, k, lambda_ = lambda_, cross_operator = cross_operator, 
                 population_size = population_size, cross_prob = cross_prob, mutation_prob = mutation_prob,
                 generation_per_ls = 10, perc_ls = 0.1, best_population = True)


