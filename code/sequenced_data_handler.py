import numpy as np
import math
import sklearn


class SequenceDataHandler():

	
	def __init__(self, sequence_length, sequence_stride, feature_size, data_scaler):


		#Public properties
		self._sequence_length = sequence_length
		self._sequence_stride = sequence_stride
		self._data_scaler = data_scaler
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None

		#Read Only properties
		self._feature_size = feature_size
		self._df_train = None
		self._df_test = None
		self._X_train_list = list()
		self._X_crossVal_list = list()
		self._X_test_list = list()
		self._y_train_list = list()
		self._y_crossVal_list = list()
		self._y_test_list = list()
		self._load_data_from_origin = True


	#def load_data():
		"""This has to be implemented in the child class"""


	def generate_train_data(self, unroll=True):
		"""Create sequenced data using sequence_length and sequence_stride"""

		self._X_train = None
		self._y_train = None

		n_m = 0
		sequences_per_sample = 	list()
		num_sequences = 0
		num_samples = len(self._X_train_list)

		#Calculate the total number of sequences based on the number of individual sequences
		for sample in self._X_train_list:
			n_m = math.floor((sample.shape[0]-self._sequence_length)/self._sequence_stride) + 1
			num_sequences = num_sequences + n_m
			sequences_per_sample.append(n_m)

		#Different shapes for the arrays depending wether data is to be unrolled or not
		if unroll == True:
			self._X_train, self._y_train = np.empty([num_sequences, self._feature_size*self._sequence_length]), np.empty([num_sequences, 1])
		else:
			self._X_train, self._y_train = np.empty([num_sequences, self._sequence_length, self._feature_size]), np.empty([num_sequences, 1])

		k = 0
		#Create the feature matrix by moving the sequence window for each sample
		for i in range(num_samples):
			for j in range(sequences_per_sample[i]):
				sequence_samples = self._X_train_list[i][j*self._sequence_stride:j*self._sequence_stride+self._sequence_length,:]

				if unroll == True:
					self._X_train[k,:] = np.squeeze(sequence_samples.reshape(1,-1)) #If I dont squeeze I may be able to get the shape needed for RNN
				else:
					self._X_train[k,:,:] = sequence_samples

				self._y_train[k,:] = self._y_train_list[i][j*self._sequence_stride+self._sequence_length-1]
				k = k + 1


	def generate_test_data(self, unroll=True):
		"""Create sequenced data using sequence_length and sequence_stride"""

		self._X_test = None
		self._y_test = None

		num_samples = len(self._X_test_list)

		#Different shapes for the arrays depending wether data is to be unrolled or not
		if unroll == True:
			self._X_test, self._y_test = np.empty([num_samples, self._feature_size*self._sequence_length]), np.empty([num_samples, 1])
		else:
			self._X_test, self._y_test = np.empty([num_samples, self._sequence_length, self._feature_size]), np.empty([num_samples, 1])

		for i in range(num_samples):
			sequence_samples = self._X_test_list[i][-self._sequence_length:,:]

			if unroll == True:
				self._X_test[i,:] = np.squeeze(sequence_samples.reshape(1,-1))
			else:
				self._X_test[i,:,:] = sequence_samples

			self._y_test[i,:] = self._y_test_list[i]

		#In case cross validation is enabled
		if len(self._X_crossVal_list) != 0:
			self.generate_crossValidation_data(unroll)
			

	def generate_crossValidation_data(self, unroll=True):
		"""Create sequenced data using sequence_length and sequence_stride"""

		self._X_crossVal = None
		self._y_crossVal = None

		num_samples = len(self._X_crossVal_list)

		#Different shapes for the arrays depending wether data is to be unrolled or not
		if unroll == True:
			self._X_crossVal, self._y_crossVal = np.empty([num_samples, self._feature_size*self._sequence_length]), np.empty([num_samples, 1])
		else:
			self._X_crossVal, self._y_crossVal = np.empty([num_samples, self._sequence_length, self._feature_size]), np.empty([num_samples, 1])

		for i in range(num_samples):
			sequence_samples = self._X_crossVal_list[i][-self._sequence_length:,:]

			if unroll == True:
				self._X_crossVal[i,:] = np.squeeze(sequence_samples.reshape(1,-1))
			else:
				self._X_crossVal[i,:,:] = sequence_samples

			self._y_crossVal[i,:] = self._y_crossVal_list[i]


	"""def scale_data(self):
		#Reescale the data using the specified data_scaler

		if self._data_scaler != None:
			X_train = self._data_scaler.fit_transform(X_train)
			X_test = self._data_scaler.transform(X_test)

			if cross_validation_ratio > 0:
				X_crossVal = self._data_scaler.transform(X_crossVal)

		self._X_train = X_train
		self._X_crossVal = X_crossVal
		self._X_test = X_test
	"""


	def print_sequence_shapes(self):
		"""Print the shapes of the sequences"""

		print("Sequence length " +str(self._sequence_length))
		print("Sequence stride " +str(self._sequence_stride))

		print("X_train len " + str(len(self._X_train_list)))
		print("X_crossVal len " + str(len(self._X_crossVal_list)))
		print("X_test len " + str(len(self._X_test_list)))
		print("y_train len " + str(len(self._y_train_list)))
		print("y_crossVal len " + str(len(self._y_crossVal_list)))
		print("y_test len " + str(len(self._y_test_list)))

		print("X_train[0]")
		print(self._X_train_list[0].shape)
		print(self._X_train_list[0])

		if len(self._X_crossVal_list) > 0:
			print("X_crossVal[0]")
			print(self._X_crossVal_list[0].shape)
			print(self._X_crossVal_list[0])
		
		print("X_test[0]")
		print(self._X_test_list[0].shape)
		print(self._X_test_list[0])

		print("y_train[0]")
		#print(self._y_train_list[0].shape)
		print(self._y_train_list[0])

		if len(self._y_crossVal_list) > 0:
			print("y_crossVal[0]")
			#print(self._y_crossVal_list[0].shape)
			print(self._y_crossVal_list[0])

		print("y_test[0]")
		#print(self._y_test_list[0].shape)
		print(self._y_test_list[0])


	def print_data(self, print_top=True):
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

		if print_top == True:
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
	def sequence_length(self):
		return self._sequence_length

	@sequence_length.setter
	def sequence_length(self, sequence_length):
		self._sequence_length = sequence_length

	@property
	def sequence_stride(self):
		return self._sequence_stride

	@sequence_stride.setter
	def sequence_stride(self, sequence_stride):
		self._sequence_stride = sequence_stride

	@property
	def data_scaler(self):
		return self._data_scaler

	@data_scaler.setter
	def data_scaler(self, data_scaler):
		self._data_scaler = data_scaler
		self._load_data_from_origin = True

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

	#ReadOnly Properties

	@property
	def feature_size(self):
		return self._feature_size
	
	@property
	def df_train(self):
		return self._df_train

	@property
	def df_test(self):
		return self._df_test

	@property
	def X_train_list(self):
		return self._X_train_list

	@property
	def X_crossVal_list(self):
		return self._X_crossVal_list

	@property
	def X_test_list(self):
		return self._X_test_list

	@property
	def y_train_list(self):
		return self._y_train_list

	@property
	def y_crossVal_list(self):
		return self._y_crossVal_list

	@property
	def y_test_list(self):
		return self._y_test_list

	@property
	def reload_data_from_origin(self):
		return self._reload_data_from_origin

