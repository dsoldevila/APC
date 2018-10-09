# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:54:15 2018
@author: David
"""

"""NOTES
    Refer codi espagueti
"""


import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt

class Regression:

    def __init__(self, X_min, X_max, target, db_col, path):
        self.X_MIN = X_min #Attributes' range of columns in DB
        self.X_MAX = X_max
        self.TARGET= target #Column index of target
        self.DB_COL = db_col
        self.loadDataset(path)
        return
        
        
    def meanSquaredError(self, y, prediction):
        """
        Calcula l'error quadràtic mitjà comés pel regressor
        @param target
        @param reg 
        """
        return ((y-prediction)**2).mean()
    
    def loadDataset(self, path):
        """
        Carrega la base de dades
        @param path (abs or not)
        """
        data = np.genfromtxt(path, delimiter=",")
        
        self.X = data[:, self.X_MIN:self.X_MAX]
        self.y = data[:, self.TARGET]
        
        self.y = self.y.reshape(self.y.shape[0], 1)     
        return
        
    def splitDataset(self, train_ratio=0.8):
        """
        Divideix aleatòriament la base de dades en training set i validation set
        """
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        n_train = int(np.floor(self.X.shape[0]*train_ratio))
        indices_train = indices[:n_train]
        indices_val = indices[-n_train:]
        X_train = self.X[indices_train, :]
        y_train = self.y[indices_train]
        X_val = self.X[indices_val, :]
        y_val = self.y[indices_val]
        
        return X_train, y_train, X_val, y_val
        
    def regression(self, X, y):
        regr = LinearRegression()
        regr.fit(X, y)
        return regr
    
    def Standarize(self):
        mean = self.X.mean(axis=0) #compute mean for every attribute
        std = self.X.std(0) #standard deviation
        X = self.X - mean
        X /= std

        return  X
    
    def Plot(self, x, x_name):
        plt.figure()
        plt.title("Histograma de l'atribut "+str(x_name))
        plt.xlabel("Attribute Value")
        plt.ylabel("Count")
        plt.hist(x[:,:], bins=11, range=[np.min(x[:,:]), np.max(x[:,:])], histtype="bar", rwidth=0.8)
        return


    
def p1a_c():
    """
    Calcula MSEs
    """
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    regr = Regression(2, 8, 8, db_col, "Database/machine.data.txt")
    
    X_t, y_t, X_v, y_v = regr.splitDataset()
    
    
    #Calculo MSEs
    mse_list = []
    
    xt = X_t[:, 0].reshape(X_t.shape[0], 1)
    xv = X_v[:, 0].reshape(X_v.shape[0], 1)
    
    linear_regression = regr.regression(xt, y_t)
    predicted = linear_regression.predict(xv)

    mse = regr.meanSquaredError(y_v, predicted)
    mse_list.append((db_col[0+regr.X_MIN], mse))
    
    lowest_mse = mse
    lowest_mse_i = 0
   
    for i in range(1, regr.X_MAX-regr.X_MIN):
        xt = X_t[:, i].reshape(X_t.shape[0], 1)
        xv = X_v[:, i].reshape(X_v.shape[0], 1)
        
        linear_regression = regr.regression(xt, y_t)
        predicted = linear_regression.predict(xv)

        mse = regr.meanSquaredError(y_v, predicted)
        mse_list.append((db_col[0+regr.X_MIN], mse))
        
        con = (lowest_mse<mse)
        lowest_mse = lowest_mse if con else mse
        lowest_mse_i = lowest_mse_i if con else i
        
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(db_col[lowest_mse_i+regr.X_MIN]+ " has the lowest mse: "+str(lowest_mse))
    
    return

def p1a_c_1():
    """
    Plot standarized attributes
    """
    
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    regr = Regression(2, 8, 8, db_col, "Database/machine.data.txt")
    
    X = regr.Standarize()
    
    for i in range(6):
        x = X[:, i].reshape(X.shape[0], 1)
        regr.Plot(x, db_col[i+regr.X_MIN])
        
    return
    
"""
p1a_c(standarize=False)
print("\n#Compute MSEs with standarized attributes:")
p1a_c(standarize=True)
print("\n#Compute MSE with top 3 attributes with lowest std:")
p1a_c2(standarize=True)
"""

print("#Compute MSEs with raw attributes:")
p1a_c()
print("#Plot standarized variables")
p1a_c_1()
