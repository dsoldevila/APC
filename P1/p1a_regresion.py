# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:54:15 2018
@author: David
"""

"""NOTES
    Refer codi espagueti
"""


#import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

class Regression:

    def __init__(self, X_min, X_max, target, db_col, path, train_ratio=0.8):
        self.X_MIN = X_min #Attributes' range of columns in DB
        self.X_MAX = X_max
        self.TARGET= target #Column index of target
        self.DB_COL = db_col
        self.loadDataset(path)
        self.splitDataset(train_ratio)
        return
        
        
    def meanSquaredError(self, y, prediction):
        """
        Calcula l'error quadràtic mitjà comés pel regressor
        @param target
        @param reg 
        """
        return ((y-prediction)**2).mean()
    
    def computeMSE(self, X_train, X_val, indices, plot):
        mse_list = []
    
        xt = X_train[:, 0].reshape(X_train.shape[0], 1)
        xv = X_val[:, 0].reshape(X_val.shape[0], 1)
        
        linear_regression = self.regression(xt, self.y_train)
        predicted = linear_regression.predict(xv)
    
        mse = self.meanSquaredError(self.y_val, predicted)
        mse_list.append((self.DB_COL[indices[0]], mse))
        
        if plot:
            self.plotR(xv, self.DB_COL[indices[0]], self.y_val, predicted)
        
        lowest_mse = mse
        lowest_mse_i = indices[0]
       
        for i in range(1, len(indices)):
            xt = X_train[:, i].reshape(X_train.shape[0], 1)
            xv = X_val[:, i].reshape(X_val.shape[0], 1)
        
            linear_regression = self.regression(xt, self.y_train)
            predicted = linear_regression.predict(xv)
    
            mse = self.meanSquaredError(self.y_val, predicted)
            mse_list.append((self.DB_COL[indices[i]], mse))
            
            if plot:
                self.plotR(xv, self.DB_COL[indices[i]], self.y_val, predicted)
            
            con = (lowest_mse<mse)
            lowest_mse = lowest_mse if con else mse
            lowest_mse_i = lowest_mse_i if con else indices[i]
            
        return mse_list, self.DB_COL[lowest_mse_i], lowest_mse
    
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
        
    def splitDataset(self, train_ratio):
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
    def plotR3D(self, X_v, predicted, x1_name, x2_name):
        # Afegim els 1's
        A = np.hstack((X_v,np.ones([X_v.shape[0],1])))
        w = np.linalg.lstsq(A,predicted, rcond=-1)[0]
        
        #Dibuixem
        #1r creem una malla acoplada a la zona de punts per tal de representar el pla
        malla = (range(20) + 0 * np.ones(20)) / 10 
        malla_x1 =  malla * (max(X_v[:,0]) - min(X_v[:,0]))/2 + min(X_v[:,0])
        malla_x2 =  malla * (max(X_v[:,1]) - min(X_v[:,1]))/2 + min(X_v[:,1])
        
        #la fucnio meshgrid ens aparella un de malla_x1 amb un de malla_x2, per atot
        #element de mallax_1 i per a tot element de malla_x2.
        xplot, yplot = np.meshgrid(malla_x1 ,malla_x2)
        
        #ara creem la superficies que es un pla
        zplot = w[0] * xplot + w[1] * yplot + w[2]
        
        #Dibuixem punts i superficie
        plt3d = plt.figure('Coeficiente prismatico -- Relacio longitud desplacament 3D').gca(projection='3d')
        plt.xlabel(x1_name)
        plt.ylabel(x2_name)
        plt3d.plot_surface(xplot,yplot,zplot, color='red')
        plt3d.scatter(X_v[:,0],X_v[:,1],self.y_val)
        
        return

def p1a_b(path):
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    x_min = 2
    x_max = 8
    target = 8
    
    print("#Trobar el millor train ratio")
    train_ratios = (0.8, 0.7, 0.6, 0.5)
    mse_list = []
    regr = Regression(x_min, x_max, target, db_col, path, train_ratio=train_ratios[0])
    X_t = regr.X_train[:, 0:3]
    X_v = regr.X_val[:, 0:3]
    linear_regression = regr.regression(X_t, regr.y_train)
    predicted = linear_regression.predict(X_v)
    mse = regr.meanSquaredError(regr.y_val, predicted)
    mse_list.append((train_ratios[0], mse))
    best_tr = train_ratios[0]
    lowest_mse = mse
    
    for tr in train_ratios[1:]:
        regr.splitDataset(train_ratio=tr)
        X_t = regr.X_train[:, 0:3]
        X_v = regr.X_val[:, 0:3]
        linear_regression = regr.regression(X_t, regr.y_train)
        predicted = linear_regression.predict(X_v)
        mse = regr.meanSquaredError(regr.y_val, predicted)
        mse_list.append((tr, mse))
        con = (lowest_mse < mse)
        best_tr = best_tr if con else tr
        lowest_mse = lowest_mse if con else mse
    print(mse_list)
    print(str(best_tr)+" is the best train ratio\n")
    
    print("#Representació 3D de la regressió dels attributs MMIN, MMAX")
    regr.splitDataset(train_ratio=best_tr)
    X_t = regr.X_train[:, 0:3]
    X_v = regr.X_val[:, 0:3]
    linear_regression = regr.regression(X_t, regr.y_train)
    predicted = linear_regression.predict(X_v)
    regr.plotR3D(X_v[:, 1:], predicted, "MMIN", "MMAX")
    return
    
def p1a_c(path):
    """
    Calcula MSEs i imprimeix les rectes de regressió per aca attribut
    """
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    x_min = 2
    x_max = 8
    target = 8
    regr = Regression(x_min, x_max, target, db_col, path)
    
    #Calculo MSEs e imprimir regresión
    print("#Compute MSEs with raw attributes and print linear regression:")
    mse_list, lm_name, lowest_mse = regr.computeMSE(regr.X_train, regr.X_val, np.arange(x_min, x_max), True)
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(lm_name+ " has the lowest mse: "+str(lowest_mse)+"\n")
    
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
    print("#Best attreibutes (MYCT, MMIN, MMAX) mse: "+str(mse))
     
    return

if __name__ == "__main__":
    #p1a_c("Database/machine.data.txt")
    p1a_b("Database/machine.data.txt")
