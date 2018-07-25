import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib
import seaborn as sns
import pandas as pd
import time
from math import sqrt

import CMAPSAuxFunctions
import custom_scores

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from . import TunableModel


class NonSequenceTunableModel(TunableModel):

		def __init__(self, modelName, model, lib_type, dataHandler, dataScaler = None, epochs = 250, batch_size = 512):

			super(TunableModel, self).(modelName, model, lib_type, epochs=epochs, batch_size=batch_size)

			#public properties
			self.dataHandler = dataHandler
			self.dataScaler = dataScaler  #Can be any scaler from scikit or using the same interface


		def loadData(verbose=0, crossValRatio=0.0):
			"""Load the data using the corresponding Data Handler, apply the corresponding scaling and reshape"""

			#A call to the Data Handler is done, DataHandler must deliver data with shape X = (samples, features), y = (samples, size_output)
			train, crossValidation, test  = DataHandler.loadData()

			#Rescale the data
			if self.dataScaler != None:
				X_train = self.dataScaler.fit_transform(X_train)
				X_test = self.dataScaler.transform(X_test)

				if crossValRatio > 0:
					X_crossVal = self.dataScaler.transform(X_crossVal)

			self.X_train = X_train
			self.X_crossVal = X_crossVal
			self.X_test = X_test
			self.y_train = np.ravel(y_train)
			self.y_crossVal = np.ravel(y_crossVal)
			self.y_test = np.ravel(y_test)


		def printData(self, printTop=True):
		"""Print the shapes of the data and the first 5 rows"""

			if self.X_train is None:
				print("No data available")
				return

			print("Printing shapes\n")
			
			print("Training data (X, y)")
			print(self.X_train.shape)
			print(self.y_train.shape)
			
			if self.X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self.X_crossVal.shape)
				print(self.y_crossVal.shape)

			print("Testing data (X, y)")
			print(self.X_test.shape)
			print(self.y_test.shape)

			if printTop == True:
				print("Printing first 5 elements\n")
				
				print("Training data (X, y)")
				print(self.X_train[:5,:])
				print(self.y_train[:5,:])

				if self.X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self.X_crossVal[:5,:])
					print(self.y_crossVal[:5,:])

				print("Testing data (X, y)")
				print(self.X_test[:5,:])
				print(self.y_test[:5,:])
			else:
				print("Printing last 5 elements\n")
				
				print("Training data (X, y)")
				print(self.X_train[-5:,:])
				print(self.y_train[-5:,:])

				if self.X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self.X_crossVal[-5:,:])
					print(self.y_crossVal[-5:,:])

				print("Testing data (X, y)")
				print(self.X_test[-5:,:])
				print(self.y_test[-5:,:])


		#Property definition

		@property
		def dataHandler(self):
			return self.__dataHandler

		@dataHandler.setter
		def dataHandler(self, dataHandler):
			self.__dataHandler = dataHandler

		@property
		def dataScaler(self):
			return self.__dataScaler

		@dataScaler.setter
		def dataScaler(self, dataScaler):
			self.__dataScaler = dataScaler



