import numpy as np
import random
import pandas as pd

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

# IP Address: 169.236.181.40
# User: dbAdmin
# Password: dbAdmin
# Database: damadics

class MNISTDataHandler():

	'''
	TODO: column information here
	'''


	#Method definition

	def __init__(self):

		#Public properties
		

		# Splitting. This is what is used to train
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None



	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, unroll=False):
		"""Unroll just to keep compatibility with the API"""


		# dataPoints = self._sqlsession.query(ValveReading)

		if verbose == 1:
			print("Loading data. Cros-Validation ratio {}".format(cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		(self._X_train, self._y_train), (self._X_test, self._y_test) = mnist.load_data()

		train_samples = self._X_train.shape[0]
		test_samples = self._X_test.shape[0]
        
		self._X_train = self._X_train.reshape(train_samples,-1)
		self._X_test = self._X_test.reshape(test_samples,-1)
        
		self._X_train = self._X_train.astype('float32')
		self._X_test = self._X_test.astype('float32')
            
		self._X_train /= 255  #scaling the features
		self._X_test /= 255
        
		self._y_train = np_utils.to_categorical(self._y_train,10)
		self._y_test = np_utils.to_categorical(self._y_test,10)

		#Create cross-validation
		if cross_validation_ratio != 0:
			self._X_train, self._X_crossVal, self._y_train, self._y_crossVal = train_test_split(self._X_train, self._y_train, train_size=1-cross_validation_ratio)

        

	#Property definition

	@property
	def X_train(self):
		return self._X_train
    
	@X_train.setter
	def X_train(self, X_train):
		self._X_train = X_train
    
	@property
	def X_crossVal(self):
		return self._X_crossVal
    
	@X_crossVal.setter
	def X_crossVal(self, X_crossVal):
		self._X_crossVal = X_crossVal
    
	@property
	def X_test(self):
		return self._X_test
    
	@X_test.setter
	def X_test(self, X_test):
		self._X_test = X_test
    
	@property
	def y_train(self):
		return self._y_train
    
	@y_train.setter
	def y_train(self, y_train):
		self._y_train = y_train
    
	@property
	def y_crossVal(self):
		return self._y_crossVal
    
	@y_crossVal.setter
	def y_crossVal(self, y_crossVal):
		self._y_crossVal = y_crossVal
    
	@property
	def y_test(self):
		return self._y_test
    
	@y_test.setter
	def y_test(self, y_test):
		self._y_test = y_test


