import numpy as np
import math


class SequenceDataHandler():

	
	def __init__(self, sequence_length, sequence_stride, feature_size):


		#Public properties
		self._sequence_length = sequence_length
		self._sequence_stride = sequence_stride

		#Read Only properties
		self._feature_size = feature_size
		self._df_train = None
		self._df_test = None
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None
		self._X_train_list = list()
		self._X_crossVal_list = list()
		self._X_test_list = list()
		self._y_train_list = list()
		self._y_crossVal_list = list()
		self._y_test_list = list()

		print("super init")


	def create_sequenced_train_data(self):
		"""Create sequenced data using sequence_length and sequence_stride"""

		n_m = 0
		sequences_per_sample = 	list()
		num_sequences = 0
		num_samples = len(self._X_train_list)

		#Calculate the total number of sequences based on the number of individual sequences
		for sample in self._X_train_list:
			n_m = math.floor((sample.shape[0]-self._sequence_length)/self._sequence_stride) + 1
			num_sequences = num_sequences + n_m
			sequences_per_sample.append(n_m)

		self._X_train, self._y_train = np.empty([num_sequences, self._feature_size*self._sequence_length]), np.empty([num_sequences, 1])

		k = 0
		#Create the feature matrix by moving the sequence window for each sample
		for i in range(num_samples):
			for j in range(sequences_per_sample[i]):
				sequence_samples = self._X_train_list[i][j*self._sequence_stride:j*self._sequence_stride+self._sequence_length,:]
				self._X_train[k,:] = np.squeeze(sequence_samples.reshape(1,-1)) #If I dont squeeze I may be able to get the shape needed for RNN
				self._y_train[k,:] = self._y_train_list[i][j*self._sequence_stride+self._sequence_length-1]
				k = k + 1


	def create_sequenced_test_data(self):
		"""Create sequenced data using sequence_length and sequence_stride"""

		num_samples = len(self._X_test_list)

		self._X_test, self._y_test = np.empty([num_samples, self._feature_size*self._sequence_length]), np.empty([num_samples, 1])

		k = 0
		for i in range(num_samples):
			sequence_samples = self._X_test_list[i][-self._sequence_length:,:]
			self._X_test[k,:] = np.squeeze(sequence_samples.reshape(1,-1))
			self._y_test[k,:] = self._y_test_list[i]
			k = k + 1

		#In case cross validation is enabled
		if len(self._X_crossVal_list) != 0:
			num_samples = len(self._X_crossVal_list)

			self._X_crossVal, self._y_crossVal = np.empty([num_samples, self._feature_size*self._sequence_length]), np.empty([num_samples, 1])

			k = 0
			for i in range(num_samples):
				sequence_samples = self._X_crossVal_list[i][-self._sequence_length:,:]
				self._X_crossVal[k,:] = np.squeeze(sequence_samples.reshape(1,-1))
				self._y_crossVal[k,:] = self._y_crossVal_list[i]
				k = k + 1


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


	def print_data(self, printTop=True):
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
	def X_train(self):
		return self._X_train

	@property
	def X_crossVal(self):
		return self._X_crossVal

	@property
	def X_test(self):
		return self._X_test

	@property
	def y_train(self):
		return self._y_train

	@property
	def y_crossVal(self):
		return self._y_crossVal

	@property
	def y_test(self):
		return self._y_test

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

