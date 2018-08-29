import math
import numpy as np
import csv
import time as tim
import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import custom_scores
from data_handler_CMAPS import CMAPSDataHandler
from tunable_model import SequenceTunableModelRegression
import CMAPSAuxFunctions

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras import regularizers
import tensorflow as tf

from scipy.optimize import differential_evolution

global l2_lambda_regularization, l1_lambda_regularization, global_counter

l2_lambda_regularization = 0.20
l1_lambda_regularization = 0.20
global_counter = 0

def create_placeholders(input_shape, output_shape):
    
    X = tf.placeholder(tf.float32, shape=(None,input_shape), name="X")
    y = tf.placeholder(tf.float32, shape=None, name="y")
    
    return X, y

def get_minibatches(X_full, y_full, batch_size, shuffle=True):
    
    full_size = X_full.shape[0]
    total_batches = math.floor(full_size/batch_size)
    remainder = full_size - total_batches*batch_size

    if shuffle == True:
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


def model_tf(X):


	l2_lambda_regularization = 0.2
	
	A1 = tf.layers.dense(X, 20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name="fc1")
	A2 = tf.layers.dense(A1, 20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name="fc2")
	y = tf.layers.dense(A2, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), kernel_regularizer=regularizers.l2(l2_lambda_regularization), name="out")

	return y


def compiled_model(input_shape, output_shape):

	tf.reset_default_graph()

	X, y = create_placeholders(input_shape, output_shape)

	y_pred = model_tf(X)
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



def run_exhaustive_search():

	global global_counter
	#Perform exhaustive search to find the optimal parameters

	epochs = 20
	batch_size = 512

	#models = {'shallow-20':RULmodel_SN_5}

	#Selected as per CNN paper
	features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 
						 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
	selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
	selected_features = list(features[i] for i in selected_indices-1)
	data_folder = '../CMAPSSData'
	num_features = len(selected_features)

	window_size = 30
	window_stride = 1
	max_rul = 125
	shape = num_features*window_size

	#maxWindowSize = {'1':30, '2':20, '3':30, '4':18}
	max_window_size = {'1':30, '2':20} #Do it only for datasets 1 and 2
	total_time = {'1':0, '2':0, '3':0, '4':0}
	results = {'1':0, '2':0, '3':0, '4':0}

	lrate = LearningRateScheduler(CMAPSAuxFunctions.step_decay)

	#Create necessary objects
	dHandler_cmaps = CMAPSDataHandler(data_folder, 1, selected_features, max_rul, window_size, window_stride)

	#model = get_compiled_model(models['shallow-20'], shape, model_type='ann')
	#tunable_model = SequenceTunableModelRegression('ModelRUL_SN_5', model, lib_type='keras', data_handler=dHandler_cmaps, epochs=20)

	min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
	#tunable_model.data_scaler = min_max_scaler

	tf.summary.FileWriter('./logs/tunable_model', K.get_session().graph)

	for dataset_number in max_window_size:
		
		print("Results for dataset "+dataset_number)
		file = open("results/MLP/exhauxtive_search_tf_"+dataset_number+".csv", "w")
		start_time = tim.clock()
		
		dHandler_cmaps.change_dataset(dataset_number)
		
		writer = csv.writer(file)
		verbose = 1
		
		for r in range(90, 141):   #Load max_rul first as it forces reloading the dataset from file
			
			verbose = 1
			dHandler_cmaps.max_rul = r
			
			for w in range(15, max_window_size[dataset_number]+1):
			
				for s in range(1,11):
					
					print("Testing for w:{}, s:{}, r:{}".format(w, s, r))
					
					#Set data parameters
					dHandler_cmaps.sequence_length = w
					dHandler_cmaps.sequence_stride = s

					dHandler_cmaps.load_data(unroll=True, verbose=1, cross_validation_ratio=0)

					#Rescale the data
					dHandler_cmaps.X_train = min_max_scaler.fit_transform(dHandler_cmaps.X_train)
					dHandler_cmaps.X_test = min_max_scaler.transform(dHandler_cmaps.X_test)

					
					input_shape = dHandler_cmaps.X_train.shape[1]
					output_shape = 1

					#Create and compile the models
					#shape = num_features*w
					model = compiled_model(input_shape, output_shape)

					#Run the model
					with tf.Session() as sess:

						train_ann_tf(sess, model, dHandler_cmaps.X_train, dHandler_cmaps.y_train, epochs=epochs, batch_size=batch_size, display_step=0)

						y_pred = predict_ann_tf(sess, model, dHandler_cmaps.X_test, dHandler_cmaps.y_test,  batch_size=batch_size)

						#print(y_pred)

					y_pred = np.ravel(y_pred)
					y_true = np.ravel(dHandler_cmaps.y_test)

					scores = evaluate_model(y_pred, y_true, metrics=['rhs', 'rmse'], round = 2)

					print(scores)

					#model = get_compiled_model(models['shallow-20'], shape, model_type='ann')

					#Add model to tunable model
					#tunable_model.change_model('ModelRUL_SN', model, 'keras')
									
					#Load the data
					#tunable_model.load_data(unroll=True, verbose=verbose, cross_validation_ratio=0)
					
					#Train and evaluate
					#tunable_model.train_model(learningRate_scheduler=lrate, verbose=0)
					#tunable_model.evaluate_model(['rhs', 'rmse'], round=2)


					#cScores = tunable_model.scores
					#rmse = math.sqrt(cScores['score_1'])
					#rmse2 = cScores['rmse']
					#rhs = cScores['rhs']
					#time = tunable_model.train_time
					
					#row = [w, s, r, rmse, rhs]
					#writer.writerow(row)
					
					#msgStr = "The model variables are " + str(x) + "\tThe scores are: [RMSE:{:.4f}, RHS:{:.4f}]\n".format(rmse, rhs)
					#file.write(msgStr)

					global_counter = global_counter + 1
					
		end_time = tim.clock()
		file.close()
		totalTime[dataset_number] = end_time - start_time


run_exhaustive_search()
#new_function()
