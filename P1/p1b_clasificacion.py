# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

# Importem el mètode per carregar la base de dades del mòdul de la pràctica 1a.
from p1a_regresion import load_dataset, split_dataset

def p1b_c():
	"""
	Funció principal per a l'apartat C de la pràctica 1b.
	"""


def split_data(x, y, train_ratio=0.7):
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

