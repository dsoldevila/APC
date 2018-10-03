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

def load_dataset(path, i_attmin, i_attmax, i_target, i_dbguess=None):
    """
    Carrega la base de dades
    @param path (abs or not)
    """
    if(os.path.isabs(path) == False):
        path = os.path.join(os.getcwd(), path)
    data = np.genfromtxt(path, delimiter=",")
    
    attributes = data[:, i_attmin:i_attmax]
    target = data[:, i_target]
    
    if(i_dbguess!=None):
        db_guess = data[:, i_dbguess]
    

    target = target.reshape(target.shape[0], 1)
    db_guess = db_guess.reshape(db_guess.shape[0], 1)
    
    return attributes, target, db_guess

def regression(x, y):
    regr = LinearRegression()
    regr.fit(x, y)
    return regr
    
    
def split_data():
    return
    
def standarize():
    return

def p1a_c():
    """
    Apartat (C) de la pràctica 1a
    """
    ATT_MIN = 2 #Attributes' range of columns in DB
    ATT_MAX = 8
    TARGET = 8 #Column index of target
    DB_GUESS = 9 #Column index of the regression guess
    
    DB_COL = ["venor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    
    attributes, target, dg= load_dataset("Database\machine.data.txt", ATT_MIN, ATT_MAX, TARGET, DB_GUESS)
    
    """
    at = attributes[:, 2].reshape(attributes.shape[0], 1)
    regr = regression(at, target)
    predicted = regr.predict(at)
    """
    mse_list = []
    at = attributes[:, 0].reshape(attributes.shape[0], 1)
    
    regr = regression(at, target)
    predicted = regr.predict(at)
    mse = mean_squared_error(target, predicted)
    mse_list.append((DB_COL[0+ATT_MIN], mse))
    
    lowest_mse = mse
    lowest_mse_i = 0
   
    for i in range(1, ATT_MAX-ATT_MIN):
        at = attributes[:, i].reshape(attributes.shape[0], 1)
        regr = regression(at, target)
        predicted = regr.predict(at)
        mse = mean_squared_error(target, predicted)
        mse_list.append((DB_COL[i+ATT_MIN], mse))
        con = (lowest_mse<mse)
        lowest_mse = lowest_mse if con else mse
        lowest_mse_i = lowest_mse_i if con else i
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(DB_COL[lowest_mse_i+ATT_MIN]+ " has the lowest mse: "+str(lowest_mse))
    
    return
    
"""
attributes, target, dg= load_dataset("Database\machine.data.txt")
at = attributes[:, 2].reshape(attributes.shape[0], 1)

print(at.shape)
print(target.shape)

regr = regression(at, target)
predicted = regr.predict(at)

plt.plot(at, predicted, 'r')
plt.scatter(at, target)

print("MSE: ")
print(mean_squared_error(target, predicted))

print("SCORE: ")
print(regr.score(at, target))
"""
p1a_c()