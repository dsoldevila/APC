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
        self.splitDataset()
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
        self.X_train = self.X[indices_train, :]
        self.y_train = self.y[indices_train]
        self.X_val = self.X[indices_val, :]
        self.y_val = self.y[indices_val]
        return
        
    def regression(self, X, y):
        regr = LinearRegression()
        regr.fit(X, y)
        return regr
    
    def Standarize(self):
        mean = self.X_train.mean(axis=0) #compute mean for every attribute
        std = self.X_train.std(0) #standard deviation
        self.X_train_std = self.X_train - mean
        self.X_train_std /= std
        
        self.X_val_std = self.X_val - mean 
        self.X_val_std /= std

        return
    
    def plotX(self, x, x_name):
        plt.figure()
        plt.title("Histograma de l'atribut " + x_name)
        plt.xlabel("Attribute Value")
        plt.ylabel("Count")
        plt.hist(x[:,:], bins=11, range=[np.min(x[:,:]), np.max(x[:,:])], histtype="bar", rwidth=0.8)
        return
    
    def plotR(self, x, x_name, y, predicted):
        plt.figure()
        plt.title("Regressió de l'atribut " + x_name)
        plt.scatter(x, y)
        plt.plot(x, predicted, color="r")
        return


    
def p1a_c():
    """
    Calcula MSEs i imprimeix les rectes de regressió per aca attribut
    """
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    regr = Regression(2, 8, 8, db_col, "Database/machine.data.txt")
    
    
    #Calculo MSEs e imprimir regresión
    mse_list = []
    
    print("#Compute MSEs with raw attributes and print linear regression:")
    xt = regr.X_train[:, 0].reshape(regr.X_train.shape[0], 1)
    xv = regr.X_val[:, 0].reshape(regr.X_val.shape[0], 1)
    
    linear_regression = regr.regression(xt, regr.y_train)
    predicted = linear_regression.predict(xv)

    mse = regr.meanSquaredError(regr.y_val, predicted)
    mse_list.append((db_col[0+regr.X_MIN], mse))
    
    regr.plotR(xv, db_col[0], regr.y_val, predicted)
    
    lowest_mse = mse
    lowest_mse_i = 0
   
    for i in range(1, regr.X_MAX-regr.X_MIN):
        xt = regr.X_train[:, i].reshape(regr.X_train.shape[0], 1)
        xv = regr.X_val[:, i].reshape(regr.X_val.shape[0], 1)
    
        linear_regression = regr.regression(xt, regr.y_train)
        predicted = linear_regression.predict(xv)

        mse = regr.meanSquaredError(regr.y_val, predicted)
        mse_list.append((db_col[0+regr.X_MIN], mse))
    
        regr.plotR(xv, db_col[0+regr.X_MIN], regr.y_val, predicted)
        
        con = (lowest_mse<mse)
        lowest_mse = lowest_mse if con else mse
        lowest_mse_i = lowest_mse_i if con else i
        
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(db_col[lowest_mse_i+regr.X_MIN]+ " has the lowest mse: "+str(lowest_mse)+"\n")
    
    
    #Plot standard attributes
    print("#Plot standarized variables\n")
    regr.Standarize()
    for i in range(6):
        x = regr.X_train_std[:, i].reshape(regr.X_train_std.shape[0], 1)
        regr.plotX(x, db_col[i+regr.X_MIN])
    #TODO Calcular MSEs con las variables normalizadas
    
    #Recompute regression with top3 attributes
    X_t = regr.X_train[:, 0:3]
    X_v = regr.X_val[:, 0:3]
    
    linear_regression = regr.regression(X_t, regr.y_train)
    predicted = linear_regression.predict(X_v)

    mse = regr.meanSquaredError(regr.y_val, predicted)
    print("#Best attreibutes mse: "+str(mse))
          
    return

p1a_c()
