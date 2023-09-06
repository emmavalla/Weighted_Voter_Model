import numpy as np
from numpy import convolve
from copy import deepcopy
import matplotlib.pyplot as plt

def make_F(r, x_values, distribution):
    types = np.size(x_values)
    F = np.zeros((types, types+1,types+1), dtype = float)
    for i in range(types):
        F[i,0,0]= r/(r+x_values[i])
        F[i,i+1,0]= -1
        for j in range(types):
            F[i,i+1,j+1]= distribution[j]*x_values[i]/(x_values[i]+r)
    return F


def make_J(F):
    types = np.shape(F)[0]
    J = np.zeros((types,types,types+1))
    for i in range(types):
        for j in range(types):
            J[i,j,:] = F[i,j+1,:] + F[i,:,j+1]
    return J


def make_current_J(J,current_estimate):
    types = np.size(current_estimate)
    current_J = np.zeros((types,types), dtype= float)
    for i in range(0,types):
        for j in range(0,types):
            current_J[i,j] = J[i, j, 0] + np.matmul(current_estimate, J[i, j, 1:(types+1)])
    return current_J


def make_current_F(F,current_estimate):
    types = np.shape(F)[0]
    estimate_array = np.ones((types+1,1),dtype = float)
    estimate_array[1:(types+1),0]=current_estimate
    estimate_mat = estimate_array*np.transpose(estimate_array)
    current_F = np.zeros((types), dtype= float)
    for i in range(0,types):
        current_F[i] = np.sum(F[i,:,:]*estimate_mat)
    return current_F

# r is defined here as  1/ (the expected value of the rates of site B)
# size-biased distrition, therefore r = mean of site B quality
# x_values ------------- (the noise distribution values) for site A
# distribution ----------- (the equivalent noise distribution probs) for site A

def find_extinction(r, x_values, distribution):
    F = make_F(r, x_values,distribution)
    J = make_J(F)
    current_estimate = np.zeros_like(x_values)
    current_estimate[:] = 0.1
    dist = 1
    while dist>0.000001:
        current_J = make_current_J(J,current_estimate)
        current_F = make_current_F(F,current_estimate)
        y = -np.matmul(np.linalg.inv(current_J),current_F)
        current_estimate = current_estimate + y
        dist = np.linalg.norm(y)
    return current_estimate
