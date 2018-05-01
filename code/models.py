import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib
import seaborn as sns
import pandas as pd
import time

import CMAPSAuxFunctions

#from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K

class tunableModel():

	maxWindowSize = {'1':30, '2':20, '3':30, '4':30}
	maxWindowStride = 10


	def __init__(self, selectedFeatures, datasetNumber = '1', constRul = 125, window_size = 20, window_stride = 1, scaler = None):
			
			#public properties
			self.selectedFeatures = selectedFeatures
			self.datasetNumber = datasetNumber
			self.constRul = constRul
			self.windowSize = window_size if window_size <= maxWindowSize[datasetNumber] else maxWindowSize[datasetNumber]
			self.windowStride = window_stride if window_stride < maxWindowSize else maxWindowSize
			self.dataScaler = scaler
			self.X_test = None
			self.X_train = None
			self.y_test = None
			self.y_train = None
			self.currentModel = None
			self.currentModelName = None

			#private properties
			self.__data_file_train = '../CMAPSSData/train_FD00'+datasetNumber+'.txt'
			self.__data_file_test = '../CMAPSSData/test_FD00'+datasetNumber+'.txt'
			self.__rul_file = '../CMAPSSData/RUL_FD00'+datasetNumber+'.txt'
			self.__models = {}
			self.__scores = {}
			self.__times = {}


	def loadData(self):
		"""This function forms the X and y matrices from the specified dataset using the rul, window size and stride specified"""

		print("Loading data for dataset " + self.datasetNumber)

		self.__data_file_train = '../CMAPSSData/train_FD00'+datasetNumber+'.txt'
		self.__data_file_test = '../CMAPSSData/test_FD00'+datasetNumber+'.txt'
		self.__rul_file = '../CMAPSSData/RUL_FD00'+datasetNumber+'.txt'

		#Get the X and y matrices with the specified time window and strides
		X_train, y_train, _ = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_train, self.selected_features, 
			time_window = self.time_window, stride = self.window_stride, dataset_type = 'train')

		X_test, _, _ = CMAPSAuxFunctions.retrieve_and_reshape_data(self.__data_file_train, self.selected_features, time_window = self.time_window, dataset_type = 'test')

		#Standardize the data
		if scaler != None:
			X_train = self.scaler.fit_transform(X_train)
			X_test = self.scaler.transform(X_test)

		y_test = np.loadtxt(rul_file)
		y_test = np.array([x if x < constRUL else constRUL for x in y_test])
		y_test = np.reshape(y_test, (y_test.shape[0], 1))

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test


	def addModel(self, modelName, model, optimizer = None, loss = None, metrics = None):
		"""Compiles the model with the specified parameters and adds it to the models list"""

		model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
		self.__models[modelName] = model


	def setCurrentModel(self, modelName):
		"""Set the current model to be operated with"""

		if modelName not in models.keys():
			print("Model name not found")
		else
			self.currentModelName = modelName
			self.currentModel = models[self.currentModelName]


	def getModelDescription(self, modelName):
		"""Provide a description of the choosen model, if no name is provided then describe the current model"""
		
		if modelName != None
			if modelName in self.__models:
				print("Description for model: " + modelName)
				self.__models[modelName].summary()
			else:
				print("Model name not found")
		else:
			print("Description for current model: " + currentModelName)
			self.currentModel.summary()

		"""plot_model(happyModel, to_file='HappyModel.png')
		SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))"""


	def getAllModelsDescription(self):
		"""Provide a description for all the models"""

		for modelName, model in self.__models:
			print("Description for model: " + modelName)
			model.summary()


	def trainCurrentModel(self):
		"""Train the current model using keras"""

		


	def getModelScores(self, modelName=None):
		"""Return the scores for the selected model, if the model was not selected return scores for all models"""

		if modelName != None:

			if modelName not in self.__models.keys():
				print("Model name not found")
				return None
			else:
				return self.__scores[modelName]
		else:
			return self.__scores

	
	def getModelTimes(self, modelName=None):	
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


	def printData(self):
		"""Print the shapes of the data and the last 5 rows"""

		if self.X_train == None:
			print("No data available")
			return

		print("Printing shapes")
		print("Training data (X, y)")
		print(self.X_train.shape)
		print(self.y_train.shape)
		print("Testing data (X, y)")
		print(self.X_test.shape)
		print(self.y_test.shape)

		print("Printing first 5 elements")
		print("Training data (X, y)")
		print(X_train[:5,:])
		print(y_train[:5,:])
		print("Testing data (X, y)")
		print(X_test[:5,:])
		print(y_test[:5,:])


	#property definition

	@property
	def selectedFeatures(self):
		return self.__selectedFeatures

	@selectedFeatures.setter(self, selectedFeatures):
		self.__selectedFeatures = selectedFeatures

	@property
	def datasetNumber(self):
		return self.__datasetNumber

	@datasetNumber.setter(self, datasetNumber):
		self.__datasetNumber = datasetNumber

	@property
	def constRul(self):
		return self.__constRul

	@constRul.setter(self, constRul):
		self.__constRul = constRul

	@property
	def windowSize(self):
		return self.__windowSize

	@windowSize.setter(self, windowSize):
		self.__windowSize = windowSize

	@property
	def windowStride(self):
		return self.__windowStride

	@windowStride.setter(self, windowStride):
		self.__windowStride = windowStride

	@property
	def dataScaler(self):
		return self.__dataScaler

	@dataScaler.setter(self, dataScaler):
		self.__dataScaler = dataScaler

	@property
	def X_test(self):
		return self.__X_test

	@X_test.setter(self, X_test):
		self.__X_test = X_test

	@property
	def X_train(self):
		return self.__X_train

	@X_train.setter(self, X_train):
		self.__X_train = X_train

	@property
	def y_test(self):
		return self.__y_test

	@y_test.setter(self, y_test):
		self.__y_test = y_test

	@property
	def y_train(self):
		return self.__y_train

	@y_train.setter(self, y_train):
		self.__y_train = y_train

	@property
	def currentModel(self):
		return self.__currentModel

	"""@currentModel.setter(self, currentModel):
		self.__currentModel = currentModel"""










