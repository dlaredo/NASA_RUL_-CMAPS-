import os
import numpy as np
import random
import pandas as pd
import time
import tensorflow as tf
from keras.callbacks import TensorBoard

from math import sqrt

import custom_scores

import CMAPSAuxFunctions

class TunableModel():


	def __init__(self, model_name, model, lib_type, epochs = 250, batch_size = 512):

			#public properties
			self._epochs = epochs
			self._batch_size = batch_size
			self._X_test = None
			self._X_train = None
			self._X_crossVal = None
			self._y_crossVal = None
			self._y_test = None
			self._y_train = None

			#ReadOnly properties
			self._lib_type = lib_type
			self._model = model
			self._model_name = model_name
			self._scores = {}
			self._train_time = None
			self._y_predicted = None



	def change_model(self, model_name, model, lib_type):
		"""Change the model"""

		self._model_name = model_name
		self._model = model
		self._lib_type = lib_type


	def get_model_description(self, plot_description = False):
		"""Provide a description of the choosen model, if no name is provided then describe the current model"""

		print("Description for model: " + self._model_name)

		if self._lib_type == 'keras':

			self._model.summary()

			if plot_description == True:
				pass
				#plot_model(happyModel, to_file='HappyModel.png')
				#SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


	def train_model(self, verbose=0, learningRate_scheduler = None, tf_session=None, tensorboard=None):
		"""Train the current model using keras/scikit"""

		startTime = time.clock()
		training_callbacks = []

		if self._lib_type == 'keras':

			if learningRate_scheduler != None:
				training_callbacks.append(learningRate_scheduler)
			#to visliaze tensorboard
			if tensorboard != None:
				training_callbacks.append(tensorboard)
				#tf.summary.FileWriter(logdir = "logs/viv_log")
			
			if self._X_crossVal is not None:
				print("training with cv")
				val_data = (self._X_crossVal, self._y_crossVal)
				self._model.fit(x = self._X_train, y = self._y_train, epochs = self._epochs, batch_size = self._batch_size, callbacks=training_callbacks, verbose=verbose, validation_data=val_data)
			else:
				print("training without cv")
				self._model.fit(x = self._X_train, y = self._y_train, epochs = self._epochs, batch_size = self._batch_size, callbacks=training_callbacks, verbose=verbose)

		elif self._lib_type == 'scikit':
			y_train = np.ravel(self._y_train)
			self._model.fit(X = self._X_train, y = self._y_train)

		elif self._lib_type == 'tensorflow':
			if tf_session == None:
				print("A valid tensorflow session is needed to perform the training")
			else:
				self.train_tf(tf_session, verbose = verbose)

		else:
			print('Library not supported')

		endTime = time.clock()

		self._train_time = endTime - startTime


	def predict_model(self, cross_validation = False, tf_session = None):
		"""Evaluate the model using the metrics specified in metrics"""

		i = 1
		default_scores = []

		if cross_validation == True:
			X_test = self._X_crossVal
			y_test = self._y_crossVal
		else:
			X_test = self._X_test
			y_test = self._y_test


		#Predict the output labels
		if self._lib_type == 'keras':
			default_scores = self._model.evaluate(x = X_test, y = y_test)
			self._y_predicted = self._model.predict(X_test)
			self._scores["loss"] = default_scores[0]
			default_scores.pop(0)
		elif self._lib_type == 'scikit':
			y_test = np.ravel(self._y_test)
			self._scores["loss"] = self.__model.score(X = X_test, y = y_test)
			self._y_predicted = self.__model.predict(X_test)
		elif self._lib_type == 'tensorflow':
			if tf_session == None:
				print("A valid tensorflow session is needed to perform the training")
			else:
				print("tensorflow test")
				self._y_predicted = self.predict_tf(tf_session)
		else:
			print('Library not supported')

		for i in range(len(default_scores)):
			self._scores["score_"+str(i+1)] =  default_scores[i]


	def train_tf(self, tf_session, verbose = 1):
		"""Function to train models in tensorflow"""

		#Retrieve model variables
		X = self._model['X_placeholder']
		y = self._model['y_placeholder']
		optimizer = self._model['optimizer']
		total_cost = self._model['total_cost']
		cost = self._model['cost']


		#To reset all variables
		tf_session.run(tf.global_variables_initializer())
		avg_cost = 0.0

		#print(self.X_train)
		#print(self.y_train)

		for epoch in range(self.epochs):

			cost_tot = 0.0
			cost_reg_tot = 0.0

			X_batches, y_batches, total_batch = CMAPSAuxFunctions.get_minibatches(self.X_train, self.y_train, self._batch_size)

			#Train with the minibatches
			for i in range(total_batch):

				batch_x, batch_y = X_batches[i], y_batches[i]

				_, c_reg, c = tf_session.run([optimizer, total_cost, cost], feed_dict={X:batch_x, y:batch_y})
				cost_tot += c
				cost_reg_tot += c_reg

			avg_cost = cost_tot/total_batch
			avg_cost_reg = cost_reg_tot/total_batch

			if verbose == 1:
				print("Epoch:", '%04d' % (epoch+1), "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost))

		print("Epoch:Final", "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost))


	def predict_tf(self, tf_session):
		"""Function to predict values using a model in tensorflow"""

		#Retrieve model variables
		X = self._model['X_placeholder']
		y = self._model['y_placeholder']
		y_pred = self._model['y_pred']

		y_pred_vector = tf_session.run([y_pred], feed_dict={X:self.X_test})

		return y_pred_vector


	#property definition

	@property
	def X_test(self):
		return self._X_test

	@X_test.setter
	def X_test(self, X_test):
		self._X_test = X_test

	@property
	def X_crossVal(self):
		return self._X_crossVal

	@X_crossVal.setter
	def X_crossVal(self, X_crossVal):
		self._X_crossVal = X_crossVal

	@property
	def X_train(self):
		return self._X_train

	@X_train.setter
	def X_train(self, X_train):
		self._X_train = X_train

	@property
	def y_test(self):
		return self._y_test

	@y_test.setter
	def y_test(self, y_test):
		self._y_test = y_test

	@property
	def y_crossVal(self):
		return self._y_crossVal

	@y_crossVal.setter
	def y_crossVal(self, y_crossVal):
		self._y_crossVal = y_crossVal

	@property
	def y_train(self):
		return self._y_train

	@y_train.setter
	def y_train(self, y_train):
		self._y_train = y_train

	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def epochs(self, epochs):
		self._epochs = epochs

	@property
	def batch_size(self):
		return self._batch_size

	@batch_size.setter
	def batch_size(self, batch_size):
		self._batch_size = batch_size


	#ReadOnlyProperties
	@property
	def model(self):
		return self._model

	@property
	def model_name(self):
		return self._model_name

	@property
	def lib_type(self):
		return self._lib_type

	@property
	def scores(self):
		return self._scores

	@property
	def train_time(self):
		return self._train_time

	@property
	def df_train(self):
		return self._df_train

	@property
	def df_test(self):
		return self._df_test

	@property
	def y_predicted(self):
		return self._y_predicted


class SequenceTunableModelRegression(TunableModel):

		def __init__(self, model_name, model, lib_type, data_handler, data_scaler = None, epochs = 250, batch_size = 512):

			super().__init__(model_name, model, lib_type, epochs=epochs, batch_size=batch_size)

			#public properties
			self._data_handler = data_handler
			self._data_scaler = data_scaler  #Can be any scaler from scikit or using the same interface

			#read only properties
			self._y_pred_rounded = None


		def load_data(self, unroll=False, cross_validation_ratio=0, verbose=0):
			"""Load the data using the corresponding Data Handler, apply the corresponding scaling and reshape"""

			#A call to the Data Handler is done, DataHandler must deliver data with shape X = (samples, features), y = (samples, size_output)
			self._data_handler.load_data(unroll=unroll, verbose=verbose, cross_validation_ratio=cross_validation_ratio)

			#Fill the arrays with the data from the DataHandler
			X_train = self._data_handler.X_train
			X_crossVal = self._data_handler.X_crossVal
			X_test = self._data_handler.X_test
			self._y_train = self._data_handler.y_train
			self._y_crossVal = self._data_handler.y_crossVal
			self._y_test = self._data_handler.y_test

			#Rescale the data

			#Implemented in the dataHandler instead due to problems with sequenced data.

			#self._data_scaler = None
			if self._data_scaler != None:
				X_train = self._data_scaler.fit_transform(X_train)
				X_test = self._data_scaler.transform(X_test)

				if cross_validation_ratio > 0:
					X_crossVal = self._data_scaler.transform(X_crossVal)

			self._X_train = X_train
			self._X_crossVal = X_crossVal
			self._X_test = X_test

			self._y_train = self._y_train
			self._y_crossVal = self._y_crossVal
			self._y_test = self._y_test

			#self._y_train = np.ravel(self._y_train)
			#self._y_crossVal = np.ravel(self._y_crossVal)
			#self._y_test = np.ravel(self._y_test)


		def evaluate_model(self, metrics=[], cross_validation = False, round = 0, tf_session=None):
			"""Evaluate the performance of the model"""

			#tf.summary.FileWriter(logdir = "logs/viv_log")
            
			self.predict_model(cross_validation = cross_validation, tf_session = tf_session)

			self._y_predicted_rounded = self._y_predicted

			if round == 1:
				self._y_predicted_rounded = np.rint(self._y_predicted_rounded)
			elif round == 2:
				self._y_predicted_rounded = np.floor(self._y_predicted_rounded)

			self._y_predicted_rounded =  np.ravel(self._y_predicted_rounded)

			"""
			if setLimits:
				y_predicted = np.clip(self.__y_pred_rounded, setLimits[0], setLimits[1])
			"""

			if cross_validation == True:
				y_true = self._y_crossVal
			else:
				y_true = self.y_test


			y_true = np.ravel(y_true)

			for metric in metrics:
				score = custom_scores.compute_score(metric, y_true, self._y_predicted_rounded)
				self._scores[metric] = score


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
				print(self._y_train[:5])

				if self._X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self._X_crossVal[:5,:])
					print(self._y_crossVal[:5])

				print("Testing data (X, y)")
				print(self._X_test[:5,:])
				print(self._y_test[:5])
			else:
				print("Printing last 5 elements\n")

				print("Training data (X, y)")
				print(self._X_train[-5:,:])
				print(self._y_train[-5:])

				if self._X_crossVal is not None:
					print("Cross-Validation data (X, y)")
					print(self._X_crossVal[-5:,:])
					print(self._y_crossVal[-5:])

				print("Testing data (X, y)")
				print(self._X_test[-5:,:])
				print(self._y_test[-5:])


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


		#Read only properties

		@property
		def y_predicted_rounded(self):
			return self._y_predicted_rounded
