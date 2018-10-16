# -*- coding: utf-8 -*-

import os
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import svm
import matplotlib.pyplot as plt

from p1a_regresion import load_dataset, split_dataset


"""
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
"""


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
	ATT_MAX = 9
	TARGET = 9 #Column index of target
	DB_GUESS = 8 #Column index of the database regression guess

	DB_COL = np.array(["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

	at_train, target_train, at_val, target_val = load_dataset(os.path.join("Database","machine.data.txt"), ATT_MIN, ATT_MAX, TARGET, DB_COL)

	# We use the Heatmap to know which variables are the most correlated
	# from the estimated performance.
	indexes = np.arange(0, ATT_MAX - ATT_MIN, dtype=int)
	df2 = pd.DataFrame(at_train[:, range(0, 7)], columns = DB_COL[indexes])
	df2['ERP'] = target_train
	corrmat = df2.corr()



def main():
	p1b_c()

if __name__ == "__main__":
	main()