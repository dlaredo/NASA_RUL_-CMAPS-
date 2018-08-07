import numpy as np
import random
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from sqlalchemy import and_
from sqlalchemy import between
from sqlalchemy.sql import exists

from sqlalchemy import desc

from datetime import datetime, timezone, timedelta

from damadicsDBMapping import *
from sequenced_data_handler import SequenceDataHandler

# IP Address: 169.236.181.40
# User: dbAdmin
# Password: dbAdmin
# Database: damadics

class ValveDataHandler(SequenceDataHandler):

	'''
	TODO: column information here
	'''


	#Method definition

	def __init__(self, start_time, end_time, selected_features, sequence_length = 1, sequence_stride = 1, data_scaler = None):

		#Public properties
		self._start_time = start_time
		self._end_time = end_time
		self._selected_features = selected_features
		self._rectify_labels = False
		self._data_scaler = data_scaler

		# Database connection
		# self._db_connection = mysql.connector.connect(user = 'root', password = 'Ying6102#', database = 'damadics')
		self._load_from_db = True

		self._column_names = {0: 'timestamp', 1: 'externalControllerOutput', 2: 'undisturbedMediumFlow', 3: 'pressureValveInlet', 4:'pressureValveOutlet',
							  5: 'mediumTemperature', 6: 'rodDisplacement', 7: 'disturbedMediumFlow', 8: 'selectedFault', 9: 'faultType', 10: 'faultIntensity'}

		# Entire Dataset
		self._df = None
		self._X = None
		self._y = None

		# Splitting. This is what is used to train
		self._df_train = None
		self._df_test = None

		#create one time session
		self._sqlsession = None

		print("init")

		#super init
		super().__init__(sequence_length, sequence_stride, len(selected_features), data_scaler)


	def connect_to_db(self,username,pasw,host,dbname):
		# self.username = username
		# self.pasw = pasw
		# self.host = host
		self.dbname = dbname
		databaseString = "mysql+mysqldb://"+username+":"+pasw+"@"+host+"/"+dbname

		self._sqlsession = None
		try:
			sqlengine = sqlalchemy.create_engine(databaseString)
			SQLSession = sessionmaker(bind=sqlengine)
			self._sqlsession = SQLSession()
			print("Connection to " + databaseString + " successfull")
		except Exception as e:
			print("e:", e)
			print("Error in connection to the database")

	def extract_data_from_db(self):

		startTime = datetime.now()

		self._df = self._sqlsession.query(ValveReading).filter(ValveReading.timestamp.between (self._start_time,self._end_time) )
		self._df = pd.read_sql(self._df.statement, self._df.session.bind)
		#dataPoints = self._sqlsession.query(exists().where(ValveReading.timestamp == '2018-07-27 15:56:22')).scalar()
		#dataPoints = self._sqlsession.query(ValveReading).order_by(ValveReading.timestamp)
		# TODO: need to check whether dataPoints is of type DataFrame. Needs to be in type DataFrame
		# TODO: check whether column names are extracted out

		# All the data with selected features is saved in this variable
		# TODO: check if self._selected_features is an array of indexes or strings
		# self._df = df.iloc[:, self._selected_features].values

		# Assumption that the output is only one column and is located at the last column out of all the selected features
		# Below if self._selected_features is an array of indexes

		column_names = ['externalControllerOutput', 'pressureValveInlet',
                'pressureValveOutlet', 'mediumTemperature','rodDisplacement', 'disturbedMediumFlow', 'selectedFault']

		self._X = self._df.loc[:, column_names[:-1]].values
		self._y = self._df.loc[:, column_names[len(column_names) - 1]].values

		# Below if self._selected_features is an array of strings
		# inputs = df.loc[:, column_names[:-1]].values
		# outputs = df.loc[:, column_names[len(column_names) - 1]].values

		# for data in self._df:
		# 	print(self._df)

		print("Extracting data from database runtime:", datetime.now() - startTime)


	def one_hot_encode(self, num_readings):

		startTime = datetime.now()

		fault_column = list()
		one_hot_matrix = np.zeros((num_readings, 20))
		fault_column = self._y

		for i in range(num_readings):
			one_hot_matrix[i, int(fault_column[i] - 1)] = 1

		print("One-hot-encoding:", datetime.now() - startTime)

		return one_hot_matrix

	# Private
	def find_samples(self, data_samples):

		'''
		Assumptions made when using this functions
		1.) The value always starts of as NOT BROKEN. First faultType value is 20.
		2.) Function is used to entire dataset and not in chunks
		'''

		# TODO: handle cases when the first readings start of as a broken value
		# TODO: ask David if he wants a minimum amount of samples in the dataset

		startTime = datetime.now()

		small_list, big_list = list(), list()
		normal_status = 20.0
		isBroken = False
		counter = 0

		for i in range(len(self._y)):
			# If True, then the current status of the valve is that it is broken
			if (isBroken):
				# The valve has been fixed and is back to its normal status
				if (self._y[i] == normal_status):
					isBroken = False
					counter += 1
					# Save everything from the small_list into the big_list
					small_list = np.vstack(small_list)
					big_list.append(small_list)
					small_list = list()
				small_list.append(data_samples[i, :])
			# The current status of the valve is that it is not broken
			else:
				if (self._y[i] != normal_status):
					isBroken = True
				# small_list = np.append(data_samples[i, :], small_list)
				small_list.append(data_samples[i, :])

		print("Splitting into samples:",datetime.now() - startTime)
		print("counter:", counter)

		return big_list, counter
    #
    #
    #
    #
    #
    #
    #
	# # Private
	# def find_samples(self, data_samples):
    #
    # '''
    # Assumptions made when using this function
    # 1.) The valve always starts of as NOT BROKEN. First faultType value is 20.
    # 2.) Function is used to entire dataset and not in chunks
    # '''
    #
	# # TODO: handle cases when the first readings starts of as a broken valve
	# # TODO: ask David if he wants a minimum amount of samples in the dataset
    #
	# small_list, big_list = list(), list()``
	# normal_status = 20.0
	# isBroken = False
	# # Counter for the number of samples there are in the dataset
	# counter = 0
    #
	# for i in range(len(self._y)):
	# 	# If True, then the current status of the valve is that it is broken
	# 	if (isBroken):
	# 		# The valve has been fixed and is back to its normal status
	# 		if (self._y[i] == normal_status):
	# 			isBroken = False
	# 			counter += 1
	# 			# Save everything from the small_list into the big_list
	# 			small_list = np.vstack(small_list)
	# 			big_list.append(small_list)
	# 			# Clear the small_list (reinitialize)
	# 			small_list = list()
	# 			small_list.append(data_samples[i, :])
    #     # The current status of the valve is that it is not broken
	# 	else:
	# 		# Broken valve discovered
	# 		if (self._y[i] != normal_status):
	# 			isBroken = True
	# 	small_list.append(data_samples[i, :])
    #
	# 	# SPECIAL CASE: the simulation does not end with a fixed valve. Therefore we shall whatever is inside the small_list and say that it is an entire sample
	# if (self._y[i] != 20):
	# 	counter += 1
	# 	small_list = np.vstack(small_list)
	# 	big_list.append(small_list)
    #
	# 	return big_list, counter


	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, test_ratio = 0, unroll = True):
		"""Load the data using the specified parameters"""

		'''
		TODO: extracting data from MySQL database using SQLALCHEMY
		Functions called here: generate_df_with_rul(self, df), generate_train_arrays(self, cross_validation_ratio = 0), generate_test_arrays(self),
							   create_sequenced_train_data(self), create_sequenced_test_data(self)


		X: df[timestamp, ..., selectedFault]
		y: df['faultType']

		'''

		# dataPoints = self._sqlsession.query(ValveReading)

		if verbose == 1:
			print("Loading data for dataset {} with window_size of {}, stride of {}. Cros-Validation ratio {}".format(self._dataset_number,
				self._sequence_length, self._sequence_stride, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if test_ratio < 0 or test_ratio > 1:
			print("Error, test ratio must be between 0 and 1")
			return

		if cross_validation_ratio + test_ratio > 1:
			print("Sum of cross validation and test ratios is greater than 1. Need to pick smaller ratios.")
			return

		if self._load_from_db == True:
			print("Loading data from database")

			# These variables are where the entire data is saved at
			self.extract_data_from_db()

			# One hot encoding
			output_one_hot_matrix = self.one_hot_encode(self._df.shape[0])

			# Finds samples within the inputs
			self._X, num_samples = self.find_samples(self._X)
			self._y, _ = self.find_samples(output_one_hot_matrix)

			# self._df_train = self.load_db_into_df(self._file_train_data)
			# self._df_test = self.load_db_into_df(self._file_test_data)
			# self._df_train, num_units, trimmed_rul_train = self.generate_df_with_rul(self._df_train)
		else:
			print("Loading data from memory")

		#Reset arrays
		"""
		self._X_train_list = list()
		self._X_crossVal_list = list()
		self._X_test_list = list()
		self._y_train_list = list()
		self._y_crossVal_list = list()
		self._y_test_list = list()
		"""

		# Split up the data into its different samples
		#Modify properties in the parent class, and let the parent class finish the data processing
		self.train_cv_test_split(cross_validation_ratio, test_ratio, num_samples)
		self.print_sequence_shapes()
		# Unroll = True for ANN
		# Unroll = False for RNN
		self.generate_train_data(unroll)
		self.generate_crossValidation_data(unroll)
		self.generate_test_data(unroll)
        #
		self._load_from_db = False # As long as the dataframe doesnt change, there is no need to reload from file

	# Private
	def train_cv_test_split(self, cross_validation_ratio, test_ratio, num_samples):
		''' From the dataframes generate the feature arrays and their labels'''

		print("split_samples num_samples:", num_samples)
		print("cross_validation_ratio:", cross_validation_ratio)
		print("test_ratio:", test_ratio)
		startTime = datetime.now()

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if test_ratio < 0 or test_ratio > 1:
			print("Error, test ratio must be between 0 and 1")
			return

		if cross_validation_ratio != 0 or test_ratio != 0:
			self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list, self._X_test_list, self._y_test_list = self.split_samples(cross_validation_ratio, test_ratio, num_samples)

		print("Train, cv, and test splitting:",datetime.now() - startTime)
		print()

	# Private
	def split_samples(self, cross_validation_ratio, test_ratio, num_samples):
		'''Split the samples according to their respective ratios'''

		shuffled_samples = list(range(0, num_samples))
		random.shuffle(shuffled_samples)

		num_crossVal = int(cross_validation_ratio * num_samples)
		#print("num_crossVal:", num_crossVal)
		num_test = int(test_ratio * num_samples)
		#print("num_test:", num_test)
		num_train = num_samples - num_crossVal - num_test
		#print("num_train:", num_train)

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()

		print(self._y[0])

		for i in range(num_train):
			#print("i:", i)
			X_train_list.append(self._X[shuffled_samples[i]])
			y_train_list.append(self._y[shuffled_samples[i]])
			# y_train_list.append(self._y[shuffled_samples[i]][-1].reshape(1, 20))
			# x = 0
			# while(len(y_train_list) == 0):
			# 	if (self._y[shuffled_samples[i]][x][19] != 1):
			# 		y_train_list.append(self._y[shuffled_samples[i]])
			# 	x += 1

			# for x in range(self._y[shuffled_samples[i]].shape[0]):
			# 	if (self._y[shuffled_samples[i]][x][19] != 1 and len(y_train_list) == 0):
			# 		y_train_list.append(self._y[shuffled_samples[i]])
			# 		print(len(y_train_list))

		for j in range(num_train, num_train + num_crossVal):
			#print("j:", j)
			X_crossVal_list.append(self._X[shuffled_samples[j]])
			y_crossVal_list.append(self._y[shuffled_samples[j]][-1].reshape(1, 20))

			# y = 0
			# while(len(y_train_list) == 0):
			# 	if (self._y[shuffled_samples[i]][y][19] != 1):
			# 		y_crossVal_list.append(self._y[shuffled_samples[i]])
			# 	y += 1

			# for y in range(self._y[shuffled_samples[j]].shape[0]):
			# 	if (self._y[shuffled_samples[j]][y][19] != 1 and len(y_crossVal_list) == 0):
			# 		y_crossVal_list.append(self._y[shuffled_samples[j]])

		for k in range(num_train + num_crossVal, num_samples):
			#print("k:", k)
			X_test_list.append(self._X[shuffled_samples[k]])
			y_test_list.append(self._y[shuffled_samples[k]][-1].reshape(1, 20))

			# z = 0
			# while(len(y_train_list) == 0):
			# 	if (self._y[shuffled_samples[i]][x][19] != 1):
			# 		y_test_list.append(self._y[shuffled_samples[i]])
			# 	z += 1


			# for z in range(self._y[shuffled_samples[k]].shape[0]):
			# 	if (self._y[shuffled_samples[k]][z][19] != 1 and len(y_test_list) == 0):
			# 		y_test_list.append(self._y[shuffled_samples[k]])

		#print("X_test_list shape:", len(X_test_list[0].shape))

		return X_train_list, y_train_list, X_crossVal_list, y_crossVal_list, X_test_list, y_test_list

	# def train_cv_test_split(self, cross_validation_ratio = 0, test_ratio = 0, num_samples):
	# 	"""From the dataframes generate the feature arrays and their labels"""
    #
	# 	'''
	# 	Functions called here: split_samples(self, df, splitting_ratio), generate_cross_validation_from_df(self, df, sequence_length)
	# 	'''
    #
	# 	X_train_list, y_train_list = list(), list()
	# 	X_crossVal_list, y_crossVal_list = list(), list()
	# 	X_test_list, y_test_list = list()
    #
	# 	if cross_validation_ratio < 0 or cross_validation_ratio > 1 :
	# 		print("Error, cross validation must be between 0 and 1")
	# 		return
    #
	# 	if test_ratio < 0 or test_ratio > 1 :
	# 		print("Error, test ratio must be between 0 and 1")
	# 		return
    #
	# 	if cross_validation_ratio != 0 or test_ratio != 0:
	# 		X_train_list, X_test_list, X_crossVal_list, y_crossVal_list, y_train_list, y_test_list = self.split_samples(cross_validation_ratio, test_ratio, num_samples)
    #
	# 	return X_train_list, y_train_list, X_crossVal_list, y_crossVal_list, X_test_list, y_test_list


	# Private
	# def split_samples(self, cross_validation_ratio, test_ratio, num_samples):
	# 	"""Split the samples according to their respective ratios"""
    #
	# 	shuffled_samples = list(range(0, num_samples))
	# 	random.shuffle(shuffled_samples)
    #
	# 	num_crossVal = int(cross_validation_ratio * num_samples)
	# 	num_test = int(test_ratio * num_samples)
	# 	num_train = num_samples - num_crossVal - num_test
    #
	# 	X_train_list, y_train_list = list(), list()
	# 	X_crossVal, y_crossVal_list = list(), list()
	# 	X_test_list, y_test_list = list(), list()
    #
	# 	for i in range(num_train):
	# 		X_train_list.append(self._X[shuffled_samples[i]])
	# 		y_train_list.append(self._y[shuffled_samples[i]])
    #
	# 	for j in range(num_train, num_train + num_crossVal):
	# 		X_crossVal.append(self._X[shuffled_samples[j]])
	# 		y_crossVal_list.append(self._y[shuffled_samples[j]])
    #
	# 	for k in range(num_train + num_crossVal, num_samples):
	# 		X_test.append(self._X[shuffled_samples[k]])
	# 		y_test_list.append(self._y[shuffled_samples[k]])
    #
	# 	return X_train_list, X_test, X_crossVal, y_crossVal_list, y_train_list, y_test





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
	def start_time(self):
		return self._start_time
	@start_time.setter
	def start_time(self,start_time):
		self._start_time = start_time

	@property
	def sqlsession(self):
		return self._sqlsession
	@sqlsession.setter
	def sqlsession(self,sqlsession):
		self._sqlsession = sqlsession

	def __str__(self):
		return "<ValveReading(timestamp='%s',externalControllerOutput='%s',undisturbedMediumFlow='%s',pressureValveInlet='%s',pressureValveOutlet='%s',mediumTemperature='%s',\
		rodDisplacement='%s',disturbedMediumFlow='%s',selectedFault='%s',faultType='%s',faultIntensity='%s')>"\
		%(str(self._timestamp),self._externalControllerOutput,self._undisturbedMediumFlow,self.pressureValveInlet,\
		self.pressureValveOutlet,self.mediumTemperature,self.rodDisplacement,self.disturbedMediumFlow,self.selectedFault,\
		self.faultType,self.faultIntensity)



# 	def selectedFeatures(self):
# 		return self._selectedFeatures
#
# 	@selectedFeatures.setter
# 	def selectedFeatures(self, selectedFeatures):
# 		self._selectedFeatures = selectedFeatures
#
# 	@property
# 	def max_rul(self):
# 		return self._max_rul
#
# 	@max_rul.setter
# 	def max_rul(self, max_rul):
# 		self._max_rul = max_rul
#
# 	@property
# 	def rectify_labels(self):
# 		return self._rectify_labels
#
# 	@rectify_labels.setter
# 	def rectify_labels(self, rectify_labels):
# 		self._rectify_labels = rectify_labels
#
# 	#ReadOnly Properties
#
# 	@property
# 	def dataset_number(self):
# 		return self._dataset_number
#
# 	@property
# 	def data_folder(self):
# 		return self._data_folder
#
# 	@property
# 	def file_train_data(self):
# 		return self._file_train_data
#
# 	@property
# 	def file_test_data(self):
# 		return self._file_test_data
#
# 	@property
# 	def file_rul(self):
# 		return self._file_rul
#
# 	@property
# 	def load_from_file(self):
# 		return self._load_from_db
#
# 	@property
# 	def column_names(self):
# 		return self._column_names
#
# 	@property
# 	def df_train(self):
# 		return self._df_train
#
# 	@property
# 	def df_test(self):
# 		return self._df_test
#
#
#
# #Auxiliary functions
#
# def compute_training_rul(df_row, *args):
# 	"""Compute the RUL at each entry of the DF"""
#
# 	max_rul = args[1]
# 	rul_vector = args[0]
# 	rul_vector_index = int(df_row['Unit Number']) - 1
#
#
# 	if max_rul > 0 and rul_vector[rul_vector_index] - df_row['Cycle'] > max_rul:
# 		return max_rul
# 	else:
# 		return rul_vector[rul_vector_index] - df_row['Cycle']
