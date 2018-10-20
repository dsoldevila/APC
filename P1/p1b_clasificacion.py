# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import pandas as pd

class Clasificacion:

	def __init__(self, path, ATT_MIN, ATT_MAX, TARGET, DB_COL):
		self.att_min = ATT_MIN
		self.att_max = ATT_MAX
		self.target = TARGET
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
		return svclin.fit(self.X_train, np.ravel(self.Y_train))


	def regression(self):
		particions = [0.5, 0.7, 0.8]

		for part in particions:
			self.split_Dataset(part)
			logireg = LogisticRegression(solver='liblinear', multi_class='ovr')
			logireg.fit(self.X_train, np.ravel(self.Y_train))

			svmModelLin = self.train_svm('linear', C=2.0)
			svmModelPoly = self.train_svm('poly', C=2.0)
			svmModelRbf = self.train_svm('rbf', C=2.0)

			print("Correct classification Logistic ", part, "%: ", logireg.score(self.X_val, self.Y_val))
			print("SVM Linear Model ", part, "%: ", svmModelLin.score(self.X_val, self.Y_val))
			print("SVM Polynomial Model ", part, "%: ", svmModelPoly.score(self.X_val, self.Y_val))
			print("SVM RBF Model ", part, "%: ", svmModelRbf.score(self.X_val, self.Y_val))
			print("\n")


			yPredLog = logireg.predict(self.X_val)
			yPredLin = svmModelLin.predict(self.X_val)
			yPredPoly = svmModelPoly.predict(self.X_val)
			yPredRbf = svmModelRbf.predict(self.X_val)
			percent_correct_log = np.mean(self.Y_val == yPredLog).astype('float32')
			percent_correct_lin = np.mean(self.Y_val == yPredLin).astype('float32')
			percent_correct_poly = np.mean(self.Y_val == yPredPoly).astype('float32')
			percent_correct_rbf = np.mean(self.Y_val == yPredRbf).astype('float32')
			print("Percent Logistic ", part, "%: ", percent_correct_log)
			print("Percent Lineal ", part, "%: ", percent_correct_lin)
			print("Percent Polynomial ", part, "%: ", percent_correct_poly)
			print("Percent RBF ", part, "%: ", percent_correct_rbf)
			print("\n")






def p1b_c(path):
	"""
	Funció principal per a l'apartat C de la pràctica 1b.
	"""

	ATT_MIN = 2 #Attributes' range of columns in DB
	ATT_MAX = 9
	TARGET = 9 #Column index of target

	DB_COL = np.array(["vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

	clsf = Clasificacion(path, ATT_MIN, ATT_MAX, TARGET, DB_COL)
	clsf.regression()









if __name__ == "__main__":
	path = os.path.join("Database", "machine.data.txt")
	p1b_c(path)