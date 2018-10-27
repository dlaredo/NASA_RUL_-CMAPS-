import numpy as np
import random
import pandas as pd

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from keras.utils import np_utils

from sequenced_data_handler import SequenceDataHandler

# IP Address: 169.236.181.40
# User: dbAdmin
# Password: dbAdmin
# Database: damadics

class MNISTDataHandler(SequenceDataHandler):

	'''
	TODO: column information here
	'''


	#Method definition

	def __init__(self,selected_features, sequence_length = 1, sequence_stride = 1, data_scaler = None):

		#Public properties
		
		# Entire Dataset
		self._df = None
		self._X = None
		self._y = None

		# Splitting. This is what is used to train
		self._df_train = None
		self._df_test = None
        
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None


		print("init")

		#super init
		super().__init__(sequence_length=sequence_length, sequence_stride=sequence_stride, feature_size=selected_features, data_scaler=data_scaler)



	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, unroll = True):


		# dataPoints = self._sqlsession.query(ValveReading)

		if verbose == 1:
			print("Loading data window_size of {}, stride of {}. Cros-Validation ratio {}".format(
				self._sequence_length, self._sequence_stride, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

        
        #self.train_cv_test_split(cross_validation_ratio)
		

		#Reset arrays
		"""
		self._X_train_list = list()
		self._X_crossVal_list = list()
		self._X_test_list = list()
		self._y_train_list = list()
		self._y_crossVal_list = list()
		self._y_test_list = list()
		"""
#        self.create_lists(cross_validation_ratio)
#        self.generate_train_data(unroll)
#        self.generate_crossValidation_data(unroll)
#        self.generate_test_data(unroll)

		(self._X_train,self._y_train),(self._X_test,self._y_test) = mnist.load_data()
        
		self._X_train = self._X_train.reshape(60000,784)
		self._X_test = self._X_test.reshape(10000,784)
        
		self._X_train = self._X_train.astype('float32')
		self._X_test = self._X_test.astype('float32')
            
		self._X_train /= 255
		self._X_test /= 255
        
		self._y_train = np_utils.to_categorical(self._y_train,10)
		self._y_test = np_utils.to_categorical(self._y_test,10)

        
        
    
#    def create_lists(self, cross_validation_ratio=0):
#        """From the dataframes create the lists containing the necessary data containing the samples"""
#        
#        #Modify properties in the parent class, and let the parent class finish the data processing
#        self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list = self.generate_train_arrays(cross_validation_ratio)
#        self._X_test_list = self.generate_test_arrays()
#        self._y_test_list = np.loadtxt(self._file_rul)




        
    
        



	#Property definition

	@property
	def df(self):
		return self._df
	@df.setter
	def df(self, df):
		self._df = df

	@property
	def X(self):
		return self.X
	@X.setter
	def X(self, X):
		self.X = X

	@property
	def y(self):
		return self._y
	@y.setter
	def df(self, y):
		self._y = y

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


