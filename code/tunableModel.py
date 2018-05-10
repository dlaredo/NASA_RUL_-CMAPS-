import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib
import seaborn as sns
import pandas as pd
import time

import CMAPSAuxFunctions
import custom_scores

from sklearn.model_selection import train_test_split

class TunableModel():

	maxWindowSize = {'1':30, '2':20, '3':30, '4':30}
	maxWindowStride = 10


	def __init__(self, modelName, model, selectedFeatures, dataFolder, datasetNumber = 1, constRul = 125, 
		window_size = 30, window_stride = 1, scaler = None, epochs = 250, batch_size = 512):

			#public properties
			self.selectedFeatures = selectedFeatures
			self.constRul = constRul
			self.windowSize = window_size
			self.windowStride = window_stride
			self.dataScaler = scaler
			self.epochs = epochs
			self.batch_size = batch_size
			self.X_test = None
			self.X_train = None
			self.X_crossVal = None
			self.y_crossVal = None
			self.y_test = None
			self.y_train = None

			#ReadOnly properties
			self.__datasetNumber = str(datasetNumber)
			self.__dataFolder = dataFolder
			self.__data_file_train = dataFolder+'/train_FD00'+self.datasetNumber+'.txt'
			self.__data_file_test = dataFolder+'/test_FD00'+self.datasetNumber+'.txt'
			self.__rul_file = dataFolder+'/RUL_FD00'+self.datasetNumber+'.txt'
			self.__model = model
			self.__modelName = modelName
			self.__scores = {}
			self.__trainTime = None


	def loadData(self, verbose=0, crossValRatio=0):
		"""This function forms the X and y matrices from the specified dataset using the rul, window size and stride specified"""
        
		if verbose == 1:
			print("Loading data for dataset {} with window_size of {}, stride of {} and constRUL of {}. Cros-Validation ratio {}".format(self.__datasetNumber, 
				self.__windowSize, self.__windowStride, self.__constRul, crossValRatio))

		if crossValRatio < 0 or crossValRatio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		#Get the X and y matrices with the specified time window and strides
		X_crossVal = None
		y_crossVal = None

		X_train, y_train, X_crossVal, y_crossVal = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_train, self.selectedFeatures, 'train',
			time_window = self.windowSize, stride = self.windowStride, crossValidationRatio = crossValRatio)

		X_test, _, _, _ = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_test, self.selectedFeatures, 'test', time_window = self.windowSize, 
			crossValidationRatio = crossValRatio)

		#Rescale the data
		if self.dataScaler != None:
			X_train = self.dataScaler.fit_transform(X_train)
			X_test = self.dataScaler.transform(X_test)

			if crossValRatio > 0:
				X_crossVal = self.dataScaler.transform(X_crossVal)

		y_test = np.loadtxt(self.__rul_file)
		y_test = np.array([x if x < self.constRul else self.constRul for x in y_test])
		y_test = np.reshape(y_test, (y_test.shape[0], 1))


		'''
		#If indicated split training data for CV
		if generateCrossValidation == True:
			X_train, X_crossVal, y_train, y_crossVal = train_test_split(X_train, y_train, test_size=crossValRatio)
		'''


		self.X_train = X_train
		self.X_crossVal = X_crossVal
		self.X_test = X_test
		self.y_train = y_train
		self.y_crossVal = y_crossVal
		self.y_test = y_test

		if verbose == 1:
			print("Data loaded for dataset " + self.datasetNumber)


	def changeModel(self, modelName, model):
		"""Change the model"""

		self.__modelName = modelName
		self.__model = model


	def changeDataset(self, datasetNumber, dataFolder=None):
		"""Change the current dataset"""

		self.__datasetNumber = str(datasetNumber)

		if dataFolder == None:
			dataFolder = self.__dataFolder

		self.__data_file_train = dataFolder+'/train_FD00'+self.datasetNumber+'.txt'
		self.__data_file_test = dataFolder+'/test_FD00'+self.datasetNumber+'.txt'
		self.__rul_file = dataFolder+'/RUL_FD00'+self.datasetNumber+'.txt'		


	def getModelDescription(self, plotDescription = False):
		"""Provide a description of the choosen model, if no name is provided then describe the current model"""
		
		print("Description for model: " + self.__modelName)
		self.__model.summary()

		if plotDescription == True:
			plot_model(happyModel, to_file='HappyModel.png')
			#SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))



	def trainModel(self, learningRateScheduler = None, verbose=0):
		"""Train the current model using keras"""

		startTime = time.clock()
		if learningRateScheduler != None:
			self.__model.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size, callbacks=[learningRateScheduler], verbose=verbose)  
		else:
			self.__model.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size, 
                                   verbose=verbose)  
		endTime = time.clock()

		self.__trainTime = endTime - startTime


	def evaluateModel(self, metrics=[], crossValidation = False):
		"""Evaluate the model using the metrics specified in metrics"""

		i = 1

		if crossValidation == True:
			X_test = self.X_crossVal
			y_test = self.y_crossVal
		else:
			X_test = self.X_test
			y_test = self.y_test

		defaultScores = self.__model.evaluate(x = X_test, y = y_test)
		y_pred = self.__model.predict(X_test)

		scores = {}
		scores['loss'] = defaultScores[0]

		for score in defaultScores[1:]:
			scores['score_' + str(i)] = score
			i = i+1

		for metric in metrics:
			scores[metric] = custom_scores.compute_score(metric, y_test, y_pred)

		self.__scores = scores


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


	#property definition

	@property
	def selectedFeatures(self):
		return self.__selectedFeatures

	@selectedFeatures.setter
	def selectedFeatures(self, selectedFeatures):
		self.__selectedFeatures = selectedFeatures

	@property
	def constRul(self):
		return self.__constRul

	@constRul.setter
	def constRul(self, constRul):
		self.__constRul = constRul

	@property
	def windowSize(self):
		return self.__windowSize

	@windowSize.setter
	def windowSize(self, windowSize):
		self.__windowSize = windowSize

	@property
	def windowStride(self):
		return self.__windowStride

	@windowStride.setter
	def windowStride(self, windowStride):
		self.__windowStride = windowStride

	@property
	def dataScaler(self):
		return self.__dataScaler

	@dataScaler.setter
	def dataScaler(self, dataScaler):
		self.__dataScaler = dataScaler

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
	def datasetNumber(self):
		return self.__datasetNumber

	@property
	def dataFolder(self):
		return self.__dataFolder

	@property
	def model(self):
		return self.__model

	@property
	def modelName(self):
		return self.__modelName

	@property
	def data_file_train(self):
		return self.__data_file_train

	@property
	def data_file_test(self):
		return self.__data_file_test

	@property
	def rul_file(self):
		return self.__rul_file

	@property
	def scores(self):
		return self.__scores

	@property
	def trainTime(self):
		return self.__trainTime










