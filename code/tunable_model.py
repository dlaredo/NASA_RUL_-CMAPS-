import numpy as np
import random
import pandas as pd
import time
from math import sqrt

import custom_scores

from sklearn.metrics import mean_squared_error


class TunableModel():


	def __init__(self, modelName, model, lib_type, epochs = 250, batch_size = 512):

			#public properties
			self.epochs = epochs
			self.batch_size = batch_size
			self.X_test = None
			self.X_train = None
			self.X_crossVal = None
			self.y_crossVal = None
			self.y_test = None
			self.y_train = None

			#ReadOnly properties
			self.__lib_type = lib_type
			self.__model = model
			self.__modelName = modelName
			self.__scores = {}
			self.__trainTime = None
			self.__df_train = None
			self.__df_test = None
			self.__y_pred = None


	
	def changeModel(self, modelName, model, lib_type):
		"""Change the model"""

		self.__modelName = modelName
		self.__model = model
		self.__lib_type = lib_type


	def getModelDescription(self, plotDescription = False):
		"""Provide a description of the choosen model, if no name is provided then describe the current model"""
		
		print("Description for model: " + self.__modelName)

		if self.__lib_type == 'keras':

			self.__model.summary()

			if plotDescription == True:
				plot_model(happyModel, to_file='HappyModel.png')
				#SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


	def trainModel(self, learningRateScheduler = None, verbose=0):
		"""Train the current model using keras/scikit"""

		startTime = time.clock()

		if self.__lib_type == 'keras':

			if learningRateScheduler != None:
				self.__model.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size, callbacks=[learningRateScheduler], verbose=verbose)  
			else:
				self.__model.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size, verbose=verbose)

		elif self.__lib_type == 'scikit':
			y_train = np.ravel(self.y_train)
			self.__model.fit(X = self.X_train, y = y_train)
		
		else:
			print('Library not supported')

		endTime = time.clock()

		self.__trainTime = endTime - startTime


	def predictModel(self, metrics=[], crossValidation = False):
		"""Evaluate the model using the metrics specified in metrics"""

		i = 1

		if crossValidation == True:
			X_test = self.X_crossVal
			y_test = self.y_crossVal
		else:
			X_test = self.X_test
			y_test = self.y_test


		#Predict the output labels
		if self.__lib_type == 'keras':
			defaultScores = self.__model.evaluate(x = X_test, y = y_test)
			self.__y_pred = self.__model.predict(X_test)
		elif self.__lib_type == 'scikit':
			y_test = np.ravel(self.y_test)
			defaultScores = self.__model.score(X = X_test, y = y_test)
			self.__y_pred = self.__model.predict(X_test)
		else:
			print('Library not supported')

	#property definition

	@property
	def X_test(self):
		return self.__X_test

	@X_test.setter
	def X_test(self, X_test):
		self.__X_test = X_test

	@property
	def X_crossVal(self):
		return self.__X_crossVal

	@X_crossVal.setter
	def X_crossVal(self, X_crossVal):
		self.__X_crossVal = X_crossVal

	@property
	def X_train(self):
		return self.__X_train

	@X_train.setter
	def X_train(self, X_train):
		self.__X_train = X_train

	@property
	def y_test(self):
		return self.__y_test

	@y_test.setter
	def y_test(self, y_test):
		self.__y_test = y_test

	@property
	def y_crossVal(self):
		return self.__y_crossVal

	@y_crossVal.setter
	def y_crossVal(self, y_crossVal):
		self.__y_crossVal = y_crossVal

	@property
	def y_train(self):
		return self.__y_train

	@y_train.setter
	def y_train(self, y_train):
		self.__y_train = y_train

	@property
	def epochs(self):
		return self.__epochs

	@epochs.setter
	def epochs(self, epochs):
		self.__epochs = epochs

	@property
	def batch_size(self):
		return self.__batch_size

	@batch_size.setter
	def batch_size(self, batch_size):
		self.__batch_size = batch_size


	#ReadOnlyProperties
	@property
	def model(self):
		return self.__model

	@property
	def modelName(self):
		return self.__modelName

	@property
	def lib_type(self):
		return self.__lib_type

	@property
	def scores(self):
		return self.__scores

	@property
	def trainTime(self):
		return self.__trainTime

	@property
	def df_train(self):
		return self.__df_train

	@property
	def df_test(self):
		return self.__df_test

	@property
	def y_pred(self):
		return self.__y_pred


class NonSequenceTunableModel(TunableModel):

		def __init__(self, model_name, model, lib_type, data_handler, data_scaler = None, epochs = 250, batch_size = 512):

			super().__init__(model_name, model, lib_type, epochs=epochs, batch_size=batch_size)

			#public properties
			self._data_handler = data_handler
			self._data_scaler = data_scaler  #Can be any scaler from scikit or using the same interface


		def load_data(self, verbose=0, cross_validation_ratio=0):
			"""Load the data using the corresponding Data Handler, apply the corresponding scaling and reshape"""

			#A call to the Data Handler is done, DataHandler must deliver data with shape X = (samples, features), y = (samples, size_output)
			self._data_handler.load_data(verbose=verbose, cross_validation_ratio=cross_validation_ratio)

			#Fill the arrays with the data from the DataHandler
			X_train = self._data_handler.X_train
			X_crossVal = self._data_handler.X_crossVal
			X_test = self._data_handler.X_test
			self._y_train = self._data_handler.y_train
			self._y_crossVal = self._data_handler.y_crossVal
			self._y_test = self._data_handler.y_test

			#Rescale the data
			if self._data_scaler != None:
				X_train = self._data_scaler.fit_transform(X_train)
				X_test = self._data_scaler.transform(X_test)

				if crossValRatio > 0:
					X_crossVal = self._data_scaler.transform(X_crossVal)

			self._X_train = X_train
			self._X_crossVal = X_crossVal
			self._X_test = X_test
			#self._y_train = np.ravel(y_train)
			#self._y_crossVal = np.ravel(y_crossVal)
			#self._y_test = np.ravel(y_test)


		def print_data(self, print_top=True):
			"""Print the shapes of the data and the first 5 rows"""

			if self._X_train is None:
				print("No data available")
				return

			print("Printing shapes\n")
			
			print("Training data (X, y)")
			print(self._X_train.shape)
			print(self._y_train.shape)
			
			if self._X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self._X_crossVal.shape)
				print(self._y_crossVal.shape)

			print("Testing data (X, y)")
			print(self._X_test.shape)
			print(self._y_test.shape)

			if print_top == True:
				print("Printing first 5 elements\n")
				
				print("Training data (X, y)")
				print(self._X_train[:5,:])
				print(self._y_train[:5,:])

				if self._X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self._X_crossVal[:5,:])
					print(self._y_crossVal[:5,:])

				print("Testing data (X, y)")
				print(self._X_test[:5,:])
				print(self._y_test[:5,:])
			else:
				print("Printing last 5 elements\n")
				
				print("Training data (X, y)")
				print(self._X_train[-5:,:])
				print(self._y_train[-5:,:])

				if self._X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self._X_crossVal[-5:,:])
					print(self._y_crossVal[-5:,:])

				print("Testing data (X, y)")
				print(self._X_test[-5:,:])
				print(self._y_test[-5:,:])


		#Property definition

		@property
		def data_handler(self):
			return self._data_handler

		@data_handler.setter
		def data_handler(self, data_handler):
			self._data_handler = data_handler

		@property
		def data_scaler(self):
			return self._data_scaler

		@data_scaler.setter
		def data_scaler(self, data_scaler):
			self._data_scaler = data_scaler



