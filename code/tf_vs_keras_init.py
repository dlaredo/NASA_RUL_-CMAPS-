import tensorflow as tf
import numpy as np
import math
import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_handler_CMAPS import CMAPSDataHandler

import CMAPSAuxFunctions
import custom_scores
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras import regularizers

from numpy.random import seed
from tensorflow import set_random_seed


def get_minibatches(X_full, y_full, batch_size):
	
	full_size = X_full.shape[0]
	total_batches = math.floor(full_size/batch_size)
	remainder = full_size - total_batches*batch_size

	X_full, y_full = sklearn.utils.shuffle(X_full, y_full)
	
	X_batches = []
	y_batches = []
	
	for i in range(total_batches):
		X_batches.append(X_full[i*batch_size:(i+1)*batch_size])
		y_batches.append(y_full[i*batch_size:(i+1)*batch_size])

	if remainder != 0:
		X_batches.append(X_full[total_batches*batch_size:])
		y_batches.append(y_full[total_batches*batch_size:])
		total_batches = total_batches+1

	return X_batches, y_batches, total_batches


def print_tf_model_weights(sess):
	for v in tf.trainable_variables():
		
		layer = sess.run(v)
		
		print(v.name)
		print(layer.shape)
		print(layer)


def print_keras_model_weights(model):

	for layer in model.layers:
		weights = layer.get_weights() # list of numpy arrays
		
		for weight in weights:
		
			print(weight.shape)
			print(weight)


def create_placeholders(input_shape, output_shape):
	
	X = tf.placeholder(tf.float32, shape=(None,input_shape), name="X")
	y = tf.placeholder(tf.float32, shape=None, name="y")
	
	return X, y


def tf_model(X):

	l2_lambda_regularization = 0.2
	
	A1 = tf.layers.dense(X, 20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=0), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name="fc1")
	
	y = tf.layers.dense(A1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=0), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name="out")
	
	return y


def keras_model(input_shape):

	l2_lambda_regularization = 0.2

	#Create a sequential model
	model = Sequential()
	
	#Add the layers for the model
	model.add(Dense(20, input_dim=input_shape, activation='relu', kernel_initializer=keras.initializers.glorot_normal(seed=0), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name='fc1'))
	model.add(Dense(1, activation='linear', kernel_initializer=keras.initializers.glorot_normal(seed=0), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name='out'))
	
	return model


def compiled_model(input_shape, output_shape):

	tf.reset_default_graph()

	X, y = create_placeholders(input_shape, output_shape)

	y_pred = tf_model(X)
	cost = tf.losses.mean_squared_error(y, y_pred)
	reg_cost = tf.losses.get_regularization_loss()
	total_cost = cost + reg_cost

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_cost)

	return {'X_placeholder':X, 'y_placeholder':y, 'y_pred':y_pred, 'cost':cost, 'total_cost':total_cost, 'optimizer':optimizer}


def train_ann_tf(tf_session, model, X_train, y_train, epochs, batch_size, display_step = 1):
	
	#Retrieve model variables
	X = model['X_placeholder']
	y = model['y_placeholder']
	optimizer = model['optimizer']
	total_cost = model['total_cost']
	cost = model['cost']

		
	#To reset all variables
	tf_session.run(tf.global_variables_initializer())
	avg_cost = 0.0
	
	#print("Model weights after variables initialization")
	#print_tf_model_weights(sess)
	
	for epoch in range(epochs):
	    
	    cost_tot = 0.0
	    cost_reg_tot = 0.0
	    
	    X_batches, y_batches, total_batch = get_minibatches(X_train, y_train, batch_size)
	    
	    #Train with the minibatches
	    for i in range(total_batch):
	        
	        batch_x, batch_y = X_batches[i], y_batches[i]
	        
	        _, c_reg, c = tf_session.run([optimizer, total_cost, cost], feed_dict={X:batch_x, y:batch_y})
	        cost_tot += c
	        cost_reg_tot += c_reg
	        
	    avg_cost = cost_tot/total_batch
	    avg_cost_reg = cost_reg_tot/total_batch
	        
	    if display_step != 0 and epoch%display_step == 0:
	        print("Epoch:", '%04d' % (epoch+1), "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost))
	            
	print("Training complete!")
	print("Epoch:Final", "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost)) 


def predict_ann_tf(tf_session, model, X_test, y_test, batch_size):

	#Retrieve model variables
	X = model['X_placeholder']
	y = model['y_placeholder']
	y_pred = model['y_pred']

	y_pred_vector = tf_session.run([y_pred], feed_dict={X:X_test})

	return y_pred_vector


def evaluate_model(y_pred, y_true, metrics=[], round = 0):
	"""Evaluate the performance of the model"""

	if round == 1:
		y_pred = np.rint(y_pred)
	elif round == 2:
		y_pred = np.floor(y_pred)

	"""
	if setLimits:               
		y_predicted = np.clip(self.__y_pred_rounded, setLimits[0], setLimits[1])
	"""

	scores = {}

	for metric in metrics:
		score = custom_scores.compute_score(metric, y_true, y_pred)
		scores[metric] = score

	return scores


def main():

	epochs = 200
	batch_size = 512
	
	#Set the seeds to 0
	#seed(0)
	#set_random_seed(0)
	
	#Clear the previous tensorflow graph
	K.clear_session()
	tf.reset_default_graph()
	
	#Create data handler
	
	#Selected as per CNN paper
	features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 
				'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
	selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
	selected_features = list(features[i] for i in selected_indices-1)
	data_folder = '../CMAPSSData'
	
	window_size = 24
	window_stride = 1
	max_rul = 129
	
	min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
	
	dHandler_cmaps = CMAPSDataHandler(data_folder, 1, selected_features, max_rul=max_rul, sequence_length=window_size, sequence_stride=window_stride)
	
	dHandler_cmaps.load_data(unroll=True, verbose=1, cross_validation_ratio=0)
	
	#Rescale the data
	dHandler_cmaps.X_train = min_max_scaler.fit_transform(dHandler_cmaps.X_train)
	dHandler_cmaps.X_test = min_max_scaler.transform(dHandler_cmaps.X_test)
	
	dHandler_cmaps.print_data()
	
	#Create and train tensorflow model
	input_shape = dHandler_cmaps.X_train.shape[1]
	output_shape = 1

	model = compiled_model(input_shape, output_shape)

	#Run the model
	with tf.Session() as sess:

		train_ann_tf(sess, model, dHandler_cmaps.X_train, dHandler_cmaps.y_train, epochs=epochs, batch_size=batch_size, display_step=1)
		y_pred = predict_ann_tf(sess, model, dHandler_cmaps.X_test, dHandler_cmaps.y_test,  batch_size=batch_size)

	y_pred = np.ravel(y_pred)
	y_true = np.ravel(dHandler_cmaps.y_test)

	scores = evaluate_model(y_pred, y_true, metrics=['rhs', 'rmse'], round = 2)

	print(y_pred)
	print(scores)
	
	#Create keras model
	K.clear_session()
	tf.reset_default_graph()
	

	lrate = LearningRateScheduler(CMAPSAuxFunctions.step_decay)
	optimizer = Adam(lr=0.001)
	lossFunction = "mean_squared_error"
	metrics = ["mse"]
	
	num_features = len(selected_features)
	input_shape = len(selected_features)*window_size
	
	keras_nn = keras_model(input_shape)
	
	#Set the seeds to 0
	#seed(0)
	#set_random_seed(0)
	
	#print("Printing model weights after creation")
	#print_keras_model_weights(keras_nn.model)
	keras_nn.compile(optimizer=optimizer, loss=lossFunction, metrics=metrics)
	
	#print("Printing model weights after compilation")
	#print_keras_model_weights(keras_nn.model)
	
	keras_nn.fit(x = dHandler_cmaps.X_train, y = dHandler_cmaps.y_train, epochs = epochs, batch_size = batch_size, verbose=1)
	y_pred = keras_nn.predict(dHandler_cmaps.X_test)

	y_pred = np.ravel(y_pred)

	scores = evaluate_model(y_pred, y_true, metrics=['rhs', 'rmse'], round = 2)

	print(y_pred)
	print(scores)


main()





