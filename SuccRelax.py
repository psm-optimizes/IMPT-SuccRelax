import gurobipy as gp
from gurobipy import Model, GRB

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import math
import time

from collections import defaultdict
from scipy.sparse import coo_matrix, hstack, vstack, eye


def SuccRelax(DoseMat, beamlets, Tmax, Tmin, M, UD, UV, iterations, patientID):
    '''
    Problem Sets and Dimensions
    '''
    # Problem sets
    Sets = {}
    Sets['B'] = range(beamlets)          
    for struct in list(DoseMat.keys()):
        print(f"Size of {struct} data:", DoseMat[struct].shape)
        Sets[f'V_{struct}'] = range(DoseMat[struct].shape[0])
    print('Sets:', Sets)

    # Problem dimensions
    dimensions = {}
    dimensions['m'] = beamlets
    for struct in list(DoseMat.keys()):
        dimensions[f'n_{struct}'] = DoseMat[struct].shape[0]
    print('Dimensions:', dimensions)

    '''
    Optimization Model
    '''
    model = Model('IMPT-MILO')

    # Variables
    x  = model.addMVar((dimensions['m'],), lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    sl = model.addMVar((dimensions['n_Target'],), lb=0.0, vtype=GRB.CONTINUOUS, name="sl")
    su = model.addMVar((dimensions['n_Target'],), lb=0.0, vtype=GRB.CONTINUOUS, name="su")

    '''
    Constriants - Algorithm version
    '''
    for struct in list(DoseMat.keys()):
        if struct == 'Target':
            eyeMat = eye(dimensions[f'n_{struct}'])
            TmaxArray = np.full(DoseMat[struct].shape[0], Tmax) if np.isscalar(Tmax) else Tmax
            TminArray = np.full(DoseMat[struct].shape[0], Tmin) if np.isscalar(Tmin) else Tmin

            model.addMConstr(hstack([DoseMat[struct], -eyeMat]), gp.concatenate((x,su), axis = 0) , '<=', TmaxArray, name= [f'MaxDoseTargetVoxel_{voxel}' for voxel in Sets[f'V_{struct}']])
            model.addMConstr(hstack([DoseMat[struct],  eyeMat]), gp.concatenate((x,sl), axis = 0) , '>=', TminArray, name= [f'MinDoseTargetVoxel_{voxel}' for voxel in Sets[f'V_{struct}']])
        else:
            rhs = np.full(DoseMat[struct].shape[0], UD[struct]) if np.isscalar(UD[struct]) else UD[struct]
            model.addMConstr(DoseMat[struct], x, '<=', rhs, name= [f'MaxDoseOarVoxel_{struct}_{voxel}' for voxel in Sets[f'V_{struct}']])

    '''
    Objective function
    '''

    model.setObjective(gp.quicksum(sl[voxel] for voxel in Sets['V_Target']) + gp.quicksum(su[voxel] for voxel in Sets['V_Target']))
    model.ModelSense = GRB.MINIMIZE

    '''
    Solve
    '''
    model.setParam('OutputFlag', 1)
    model.setParam('LogFile', f'GRB_log_{patientID}.txt')
    model.setParam('LogToConsole',1)
    model.setParam('Crossover',0)
    model.setParam('Method',2)

    model.optimize()

    '''
    Algorithm Steps
    '''
    for iter in range(1, iterations):
        last_obj_value = model.ObjVal
        print('###################################################################################')
        print(f'################################## Iteration ({iter}) ##################################')
        print('###################################################################################')
        print('Incumbent:', last_obj_value)
        
        # Create the dictionary of shadow prices
        Shadow_Prices_Dict = {}
        for struct in list(DoseMat.keys()):
            if struct != 'Target':
                Shadow_Prices = []
                for voxel in Sets[f'V_{struct}']:
                    constr = model.getConstrByName(f'MaxDoseOarVoxel_{struct}_{voxel}')
                    Shadow_Prices.append((constr.ConstrName, constr.Pi))
                Shadow_Prices_Dict[struct] = sorted(Shadow_Prices, key=lambda x: x[1], reverse=False)
        
        # Select the top constraints based on their shadow prices and change their RHS 
        for struct in list(DoseMat.keys()):
            if struct != 'Target':
                print(f'No. of {struct} voxels to be relaxed:', math.floor(UV[struct]*dimensions[f'n_{struct}']))
                for constr_name, _ in Shadow_Prices_Dict[struct][ : math.floor(UV[struct]*dimensions[f'n_{struct}'])]: 
                    constr = model.getConstrByName(constr_name)
                    constr.setAttr(GRB.Attr.RHS, M)
                    model.update()
                for constr_name, _ in Shadow_Prices_Dict[struct][math.floor(UV[struct]*dimensions[f'n_{struct}']) : ]: 
                    constr = model.getConstrByName(constr_name)
                    constr.setAttr(GRB.Attr.RHS, UD[struct])
                    model.update()

        '''
        Solve
        '''
        model.reset()
        model.setParam('Crossover',0)
        model.setParam('Method',2)
        
        model.optimize()

        intensityVals = np.array([x[i].X for i in Sets['B']]).reshape(-1,1)

    return intensityVals

