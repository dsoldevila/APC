# -*- coding: utf-8 -*-

import os
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd

class Clasificacion:

	def __init__(self, path, ATT_MIN, ATT_MAX, TARGET, DB_GUESS, DB_COL):
		self.att_min = ATT_MIN
		self.att_max = ATT_MAX
		self.target = TARGET
		self.db_guess = DB_GUESS
		self.db_col = DB_COL
		self.load_Dataset(path)

	def load_Dataset(self, path):
		data = np.genfromtxt(path, delimiter=",")

		self.X = data[:, self.att_min:self.att_max]
		self.Y = data[:, self.target]

		self.Y = self.Y.reshape(self.Y.shape[0], 1)     

		return

	def split_Dataset(self, train_ratio):
		indices = np.arange(self.X.shape[0])
		np.random.shuffle(indices)
		n_train = int(np.floor(self.X.shape[0] * train_ratio))
		indices_train = indices[:n_train]
		indices_val = indices[-n_train:]
		self.X_train = self.X[indices_train, :]
		self.Y_train = self.Y[indices_train]
		self.X_val = self.X[indices_val, :]
		self.Y_val = self.Y[indices_val]

		return


	def train_svm(self, kernel='linear', C=0.01, gamma=0.001, probability=True):
		if(kernel =='linear'):
			svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

		if(kernel =='poly'):
			svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

		if(kernel =='rbf'):
			svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

		# l'entrenem
		return svclin.fit(self.X_train, self.Y_train)


def p1b_c(path):
	"""
	Funció principal per a l'apartat C de la pràctica 1b.
	"""

	ATT_MIN = 2 #Attributes' range of columns in DB
	ATT_MAX = 9
	TARGET = 9 #Column index of target
	DB_GUESS = 8 #Column index of the database regression guess

	DB_COL = np.array(["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

	clsf = Clasificacion(path, ATT_MIN, ATT_MAX, TARGET, DB_GUESS, DB_COL)

	clsf.split_Dataset(0.7)

	clsf.train_svm('linear')





if __name__ == "__main__":
    path = os.path.join("Database", "machine.data.txt")
    p1b_c(path)