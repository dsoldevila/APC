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
from mpl_toolkits.mplot3d import axes3d, Axes3D

class Data:

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
        data = LinearRegression()
        data.fit(X, y)
        return data
    
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
        
        #plt3d.view_init(45, 135)
        """for angle in range(0, 360):
            plt3d.view_init(30, angle)
            plt.draw()
            plt.pause(.001)"""
        
        return

class Regressor(object):
    def __init__(self, theta0, theta1, alpha, x, y):
        # Inicialitzem theta0 i theta1
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha
        self.x = x
        self.y = y
        return
    
    def predict(self, x):
        # implementar aqui la funció de prediccio
        return x*self.theta1 + self.theta0
        
    
    def __update(self, hy, y):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real.
        me = (hy-y).mean()
        theta0 = self.theta0 - self.alpha*me
        self.theta0 = theta0
        mex = ((hy-y)*self.x).mean()
        theta1 = self.theta1 - self.alpha*mex
        self.theta1 = theta1
        return
    
    def train(self, max_iter, epsilon):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        
        prediction = self.predict(self.x)
        mse = ((self.y-prediction)**2).mean()
        self.mse_list = [mse]
        i = 0
        while(mse>epsilon and i < max_iter):
            self.__update(prediction, self.y)
            prediction = self.predict(self.x)
            mse = ((self.y-prediction)**2).mean()/2
            self.mse_list.append(mse)
            i = i+1
        return
    
    

    
def p1a_c(path):
    """
    Calcula MSEs i imprimeix les rectes de regressió per cada attribut
    """
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    x_min = 2
    x_max = 8
    target = 8
    data = Data(x_min, x_max, target, db_col, path)
    
    #Calculo MSEs e imprimir regresión
    print("#Compute MSEs with raw attributes and print linear regression:")
    mse_list, lm_name, lowest_mse = data.computeMSE(data.X_train, data.X_val, np.arange(x_min, x_max), True)
    print("MSEs:")
    print(str(mse_list)+"\n")
    print(lm_name+ " has the lowest mse: "+str(lowest_mse)+"\n")
    
    #Plot standard attributes
    print("#Plot standarized variables\n")
    data.Standarize()
    for i in range(6):
        x = data.X_train_std[:, i].reshape(data.X_train_std.shape[0], 1)
        data.plotX(x, db_col[i+data.X_MIN])
    
    #Recompute regression with top 2 attributes
    X_t = data.X_train[:, 1:3]
    X_v = data.X_val[:, 1:3]
    
    linear_regression = data.regression(X_t, data.y_train)
    predicted = linear_regression.predict(X_v)

    mse = data.meanSquaredError(data.y_val, predicted)
    print("#Best attreibutes (MMIN, MMAX) mse: "+str(mse))
     
    return

def p1a_b(path):
    """
    Experimentar amb diferents training ratios. Representació 3D de la regressió
    """
    
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    x_min = 2
    x_max = 8
    target = 8
    
    print("#Trobar el millor train ratio")
    train_ratios = (0.8, 0.7, 0.6, 0.5)
    mse_list = []
    data = Data(x_min, x_max, target, db_col, path, train_ratio=train_ratios[0])
    X_t = data.X_train[:, 1:3]
    X_v = data.X_val[:, 1:3]
    linear_regression = data.regression(X_t, data.y_train)
    predicted = linear_regression.predict(X_v)
    mse = data.meanSquaredError(data.y_val, predicted)
    mse_list.append((train_ratios[0], mse))
    best_tr = train_ratios[0]
    lowest_mse = mse
    
    for tr in train_ratios[1:]:
        data.splitDataset(train_ratio=tr)
        X_t = data.X_train[:, 0:3]
        X_v = data.X_val[:, 0:3]
        linear_regression = data.regression(X_t, data.y_train)
        predicted = linear_regression.predict(X_v)
        mse = data.meanSquaredError(data.y_val, predicted)
        mse_list.append((tr, mse))
        con = (lowest_mse < mse)
        best_tr = best_tr if con else tr
        lowest_mse = lowest_mse if con else mse
    print(mse_list)
    print(str(best_tr)+" is the best train ratio\n")
    
    print("#Representació 3D de la regressió dels attributs MMIN, MMAX")
    data.splitDataset(train_ratio=best_tr)
    X_t = data.X_train[:, 1:3]
    X_v = data.X_val[:, 1:3]
    linear_regression = data.regression(X_t, data.y_train)
    predicted = linear_regression.predict(X_v)
    data.plotR3D(X_v, predicted, "MMIN", "MMAX")
    return


def p1a_a(path):
    """
    Implementar descens de gradient
    """
    db_col = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    x_min = 2
    x_max = 8
    target = 8
    alpha = [0.01, 0.1, 1]
    mse_max_iter = 500
    epsilon = 2000
    
    data = Data(x_min, x_max, target, db_col, path)
    
    data.Standarize()
    xt = data.X_train_std[:, 2].reshape(data.X_train_std.shape[0], 1)
    xv = data.X_val_std[:, 2].reshape(data.X_val_std.shape[0], 1)
    
    regr = Regressor(1, 1, alpha[0], xt, data.y_train)
    regr.train(mse_max_iter, epsilon)
    mses = np.array(regr.mse_list)
    for a in alpha[1:]:
        regr = Regressor(1, 1, a, xt, data.y_train)
        regr.train(mse_max_iter, epsilon)
        mses = np.vstack((mses, regr.mse_list))
    mses = np.transpose(mses)
    mse_max_iter = np.arange(mses.shape[0])
    plt.figure()
    plt.title("Plotting of MSE over time")
    for i in range(len(alpha)):
        plt.plot(mse_max_iter, mses[:,i], label=str(alpha[i]))
        print("Amb Learning rate = "+str(alpha[i])+", MSE = "+str(np.amin(mses[:, i])))
    #plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.show()
    #prediction = regr.predict(xv)
    #data.plotR(xv, db_col[2+x_min], data.y_val, prediction)
    
    
    return

def p1a_a_test():
    
    data = np.genfromtxt(path, delimiter=",")
    x = data[:, 4]
    y = data[:, 8] 
    
    #x = np.arange(10)
    #y = x*x
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*0.8))
    indices_train = indices[:n_train]
    indices_val = indices[-n_train:]
    xt = x[indices_train]
    yt = y[indices_train]
    xv = x[indices_val]
    yv = y[indices_val]
    
    xt = xt.reshape(xt.shape[0], 1)
    xv = xv.reshape(xv.shape[0], 1)
    yt = yt.reshape(yt.shape[0], 1)
    yv = yv.reshape(yv.shape[0], 1)
    
    mean = xt.mean() #compute mean for every attribute
    std = xt.std(0) #standard deviation
    xt = xt - mean
    xt /= std
    
    xv = xv - mean 
    xv /= std
    
    regr = Regressor(1, 1, 0.01, xt, yt)
    regr.train(1000, 2000)
    regr.plotMSE()
    prediction = regr.predict(xv)
    plt.figure()
    plt.title("Regression")
    plt.scatter(xv, yv)
    plt.plot(xv, prediction, color="r")
    
    return

if __name__ == "__main__":
    path = os.path.join("Database", "machine.data.txt")
    #p1a_c(path)
    #p1a_b(path)
    p1a_a(path)
    #p1a_a_test()
