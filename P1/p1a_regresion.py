# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:54:15 2018

@author: David
"""

"""NOTES

    1- He obviat marca i model, ja que np.genfromtxt no suporta strings
    i no sembla que tingui sentit tenir-los en compte com a mètrica de
    rendiment. Mirar si tenen correlació, un "no crec" no és vàlid


"""


import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt



def mean_squared_error(target, reg):
    """
    Calcula l'error quadràtic mitjà comés pel regressor
    @param target
    @param reg 
    """
    return ((target - reg)**2).mean()

def load_dataset(path):
    """
    Carrega la base de dades
    @param path (abs or not)
    """
    if(os.path.isabs(path) == False):
        path = os.path.join(os.getcwd(), path)
    data = np.genfromtxt(path, delimiter=",")
    
    attributes = data[:, 2:8]
    goal = data[:, 8]
    db_guess = data[:, 9]
    

    goal = goal.reshape(goal.shape[0], 1)
    db_guess = db_guess.reshape(db_guess.shape[0], 1)
    
    return attributes, goal, db_guess

def regression(x, y):
    regr = LinearRegression()
    regr.fit(x, y)
    return regr
    
    
def split_data():
    return
    
def standarize():
    return
    

attributes, goal, dg= load_dataset("Database\machine.data.txt")
at = attributes[:, 2].reshape(attributes.shape[0], 1)

print(at.shape)
print(goal.shape)

regr = regression(at, goal)
predicted = regr.predict(at)

plt.plot(at, predicted, 'r')
plt.scatter(at, goal)

print("MSE: ")
print(mean_squared_error(goal, predicted))

print("SCORE: ")
print(regr.score(at, goal))