# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:54:15 2018

@author: David
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
    @param path
    """
    if(os.path.isabs(path) == False):
        path = os.path.join(os.getcwd(), path)
    
    return
    
def split_data():
    return
    
def standarize():
    return
    