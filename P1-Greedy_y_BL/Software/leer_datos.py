# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:30:26 2020

@author: jorge
"""
import numpy as np

#Funciones para leer los datos
def read_dat(fname, delimiter = ','):
    return np.loadtxt(fname, delimiter = delimiter)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def read_constraints_matrix(fname, delimiter = ','):
    const_matrix = np.loadtxt(fname, delimiter = delimiter)
    if not check_symmetric(const_matrix):
        print("The matrix is not symmetric, there should be an error in the text file.")
        return None
    else:
        return const_matrix
    
def constraints_matrix_to_list(const_matrix):
    if const_matrix.shape[0] != const_matrix.shape[1]:
        print("This is not a square matrix.")
        return None
    return np.array([(i, j, const_matrix[i,j]) for i in range(const_matrix.shape[0]) \
                    for j in range(i+1, const_matrix.shape[1]) if const_matrix[i,j] != 0], dtype = int)