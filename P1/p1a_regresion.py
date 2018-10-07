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

def load_dataset(path, i_attmin, i_attmax, i_target, i_dbguess=None, standarize=False):
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
    #db_guess = db_guess.reshape(db_guess.shape[0], 1)
    
    return split_dataset(attributes, target, standarize=standarize)
    

def split_dataset(attributes, target, train_ratio=0.8, standarize=False):
    """
    Divideix aleatòriament la base de dades en training set i validation set
    """
    indices = np.arange(attributes.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(attributes.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[-n_train:]
    at_train = attributes[indices_train, :]
    target_train = target[indices_train]
    at_val = attributes[indices_val, :]
    target_val = target[indices_val]
    
    if(standarize):
        at_train, at_val = Standarize(at_train, at_val)
    
    return at_train, target_train, at_val, target_val

def regression(x, y):
    regr = LinearRegression()
    regr.fit(x, y)
    return regr
    
    
def Standarize(at_train, at_val):
    mean = at_train.mean(axis=0) #compute mean for every attribute
    std = at_train.std(0) #standard deviation
    at_t = at_train - mean
    at_t /= std
    
    at_v = at_val - mean
    at_v /= std
    
    """
    for i in range(6):
        plt.figure()
        plt.title("Histograma de l'atribut 0")
        plt.xlabel("Attribute Value")
        plt.ylabel("Count")
        hist = plt.hist(at_t[:,i], bins=11, range=[np.min(at_t[:,i]), np.max(at_t[:,i])], histtype="bar", rwidth=0.8)
    """
    return at_t, at_v
    

def p1a_c(standarize):
    """
    Apartat (C) de la pràctica 1a.
    Imprimeix l'atribut amb l'error quadràtic mitjà (mse) més baix, juntament amb el valor mse
    """
    ATT_MIN = 2 #Attributes' range of columns in DB
    ATT_MAX = 8
    TARGET = 8 #Column index of target
    DB_GUESS = 9 #Column index of the database regression guess
    
    DB_COL = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    
    at_train, target_train, at_val, target_val = load_dataset(os.path.join("Database","machine.data.txt"), ATT_MIN, ATT_MAX, TARGET, standarize=standarize)

    #Calculo MSEs
    mse_list = []
    at_t = at_train[:, 0].reshape(at_train.shape[0], 1)
    at_v = at_val[:, 0].reshape(at_val.shape[0], 1)
    
    regr = regression(at_t, target_train)
    
    predicted = regr.predict(at_v)

    mse = mean_squared_error(target_val, predicted)
    mse_list.append((DB_COL[0+ATT_MIN], mse))
    
    lowest_mse = mse
    lowest_mse_i = 0
   
    for i in range(1, ATT_MAX-ATT_MIN):
        at_t = at_train[:, i].reshape(at_train.shape[0], 1)
        at_v = at_val[:, i].reshape(at_val.shape[0], 1)
        regr = regression(at_t, target_train)
        predicted = regr.predict(at_v)
        
        mse = mean_squared_error(target_val, predicted)
        mse_list.append((DB_COL[i+ATT_MIN], mse))
        
        con = (lowest_mse<mse)
        lowest_mse = lowest_mse if con else mse
        lowest_mse_i = lowest_mse_i if con else i
        
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(DB_COL[lowest_mse_i+ATT_MIN]+ " has the lowest mse: "+str(lowest_mse))
    
    return
    