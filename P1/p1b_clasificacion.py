# -*- coding: utf-8 -*-

import os
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

# Importem el mètode per carregar la base de dades del mòdul de la pràctica 1a.
from p1a_regresion import load_dataset, split_dataset

def split_data(x, y, train_ratio=0.8):
	indices = np.arange(x.shape[0])
	np.random.shuffle(indices)
	n_train = int(np.floor(x.shape[0]*train_ratio))

	indices_train = indices[:n_train]
	indices_val = indices[-n_train:]

	x_train = x[indices_train, :]
	y_train = y[indices_train]

	x_val = x[indices_val, :]
	y_val = y[indices_val]
	
	return x_train, y_train, x_val, y_val

def train_svm(x, y, kernel='linear', C=0.01, gamma=0.001, probability=True):
    if(kernel =='linear'):
        svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

    if(kernel =='poly'):
        svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

    if(kernel =='rbf'):
        svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

    # l'entrenem
    return svclin.fit(x, y)


def p1b_c():
	"""
	Funció principal per a l'apartat C de la pràctica 1b.
	"""
	ATT_MIN = 2 #Attributes' range of columns in DB
	ATT_MAX = 8
	TARGET = 8 #Column index of target
	DB_GUESS = 9 #Column index of the database regression guess

	DB_COL = ["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]

	at_train, target_train, at_val, target_val = load_dataset(os.path.join("Database","machine.data.txt"), ATT_MIN, ATT_MAX, TARGET)

	x_train, y_train, x_val, y_val = split_data(at_train, at_val, 0.7)

	logReg = LogisticRegression()

	class_train = x_train[:,-1]
	class_val = x_val[:,-1]

	logReg.fit(x_train[:, :-1], class_val)

	prediction = logReg.predict(x_train[:, :-1])
	
	print(metrics.classification_report(y_true=class_val, y_pred=prediction))
	print(pd.crosstab(x_train[:,-1], prediction, rownames=['REAL'], colnames=['PREDICCION']))



def main():
	p1b_c()

if __name__ == "__main__":
	main()