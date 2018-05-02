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

class TunableModel():

	maxWindowSize = {'1':30, '2':20, '3':30, '4':30}
	maxWindowStride = 10


	def __init__(self, selectedFeatures, datasetNumber = '1', constRul = 125, window_size = 30, window_stride = 1, scaler = None, epochs = 250, batch_size = 512):
			
			#public properties
			self.selectedFeatures = selectedFeatures
			self.datasetNumber = datasetNumber
			self.constRul = constRul
			self.windowSize = window_size
			self.windowStride = window_stride
			self.dataScaler = scaler
			self.epochs = epochs
			self.batch_size = batch_size
			self.X_test = None
			self.X_train = None
			self.y_test = None
			self.y_train = None

			#private properties
			self.__data_file_train = '../CMAPSSData/train_FD00'+self.datasetNumber+'.txt'
			self.__data_file_test = '../CMAPSSData/test_FD00'+self.datasetNumber+'.txt'
			self.__rul_file = '../CMAPSSData/RUL_FD00'+self.datasetNumber+'.txt'
			self.__models = {}
			self.__scores = {}
			self.__times = {}
			self.__currentModel = None
			self.__currentModelName = None


	def loadData(self):
		"""This function forms the X and y matrices from the specified dataset using the rul, window size and stride specified"""

		print("Loading data for dataset " + self.datasetNumber)

		self.__data_file_train = '../CMAPSSData/train_FD00'+self.datasetNumber+'.txt'
		self.__data_file_test = '../CMAPSSData/test_FD00'+self.datasetNumber+'.txt'
		self.__rul_file = '../CMAPSSData/RUL_FD00'+self.datasetNumber+'.txt'

		#Get the X and y matrices with the specified time window and strides
		X_train, y_train, _ = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_train, self.selectedFeatures, 
			time_window = self.windowSize, stride = self.windowStride, dataset_type = 'train')

		X_test, _, _ = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_test, self.selectedFeatures, time_window = self.windowSize, dataset_type = 'test')

		#Standardize the data
		if self.dataScaler != None:
			X_train = self.dataScaler.fit_transform(X_train)
			X_test = self.dataScaler.transform(X_test)

		y_test = np.loadtxt(self.__rul_file)
		y_test = np.array([x if x < self.constRul else self.constRul for x in y_test])
		y_test = np.reshape(y_test, (y_test.shape[0], 1))

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test


	def addModel(self, modelName, model):
		"""Adds the model to the models list"""

		self.__models[modelName] = model

		#If this is the first model added, make it current
		if self.__currentModelName == None:
			self.__currentModelName = modelName
			self.__currentModel = self.__models[modelName]



	def setCurrentModel(self, modelName):
		"""Set the current model to be operated with"""

		if modelName not in self.__models.keys():
			print("Model name not found")
		else:
			self.__currentModelName = modelName
			self.__currentModel = self.__models[self.__currentModelName]


	def getModelDescription(self, modelName = None):
		"""Provide a description of the choosen model, if no name is provided then describe the current model"""
		
		if modelName != None:
			if modelName in self.__models.keys():
				print("Description for model: " + modelName)
				self.__models[modelName].summary()
			else:
				print("Model name not found")
		else:
			print("Description for current model: " + self.__currentModelName)
			self.__currentModel.summary()

		"""plot_model(happyModel, to_file='HappyModel.png')
		SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))"""


	def getAllModelsDescription(self):
		"""Provide a description for all the models"""

		for modelName, model in self.__models.items():
			print("Description for model: " + modelName)
			model.summary()


	def trainCurrentModel(self, learningRateScheduler = None):
		"""Train the current model using keras"""

		startTime = time.clock()
		if learningRateScheduler != None:
			self.__currentModel.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size, callbacks=[learningRateScheduler])  
		else:
			self.__currentModel.fit(x = self.X_train, y = self.y_train, epochs = self.epochs, batch_size = self.batch_size)  
		endTime = time.clock()

		self.__times[self.__currentModelName] = endTime - startTime


	def evaluateCurrentModel(self, metrics):
		"""Evaluate the model using the metrics specified in metrics"""

		i = 1

		defaultScores = self.__currentModel.evaluate(x = self.X_test, y = self.y_test)
		y_pred = self.__currentModel.predict(self.X_test)

		scores = {}
		scores['loss'] = defaultScores[0]

		for score in defaultScores[1:]:
			scores['score_' + str(i)] = score
			i = i+1

		for metric in metrics:
			scores[metric] = custom_scores.compute_score(metric, self.y_test, y_pred)

		self.__scores[self.__currentModelName] = scores


	def getModelScores(self, modelName = None):
		"""Return the scores for the selected model, if the model was not selected return scores for all models"""

		if modelName != None:

			if modelName not in self.__models.keys():
				print("Model name not found")
				return None
			else:
				return self.__scores[modelName]
		else:
			return self.__scores

	
	def getModelTimes(self, modelName = None):	
		"""Return the times for the selected model, if the model was not selected return time for all models"""

		if modelName != None:

			if modelName not in self.__times.keys():
				print("Model name not found")
				return None
			else:
				return self.__times[modelName]
		else:
			return self.__times


	def getModel(self, modelName):
		"""Retrieve a model from the models list"""

		if modelName not in self.__models.keys():
			print("Model name not found")
			return None

		return self.__models[modelName]


	def printData(self, printTop=True):
		"""Print the shapes of the data and the first 5 rows"""

		if self.X_train is None:
			print("No data available")
			return

		print("Printing shapes")
		print("Training data (X, y)")
		print(self.X_train.shape)
		print(self.y_train.shape)
		print("Testing data (X, y)")
		print(self.X_test.shape)
		print(self.y_test.shape)

		if printTop == True:
			print("Printing first 5 elements")
			print("Training data (X, y)")
			print(self.X_train[:5,:])
			print(self.y_train[:5,:])
			print("Testing data (X, y)")
			print(self.X_test[:5,:])
			print(self.y_test[:5,:])
		else:
			print("Printing last 5 elements")
			print("Training data (X, y)")
			print(self.X_train[-5:,:])
			print(self.y_train[-5:,:])
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
	def datasetNumber(self):
		return self.__datasetNumber

	@datasetNumber.setter
	def datasetNumber(self, datasetNumber):
		self.__datasetNumber = datasetNumber

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
	def y_train(self):
		return self.__y_train

	@y_train.setter
	def y_train(self, y_train):
		self.__y_train = y_train

	@property
	def currentModel(self):
		return self.__currentModel

	"""@currentModel.setter
	def currentModel(self, currentModel):
		self.__currentModel = currentModel"""

	@property
	def currentModelName(self):
		return self.__currentModelName

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










