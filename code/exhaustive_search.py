import math
import numpy as np
import csv
import time as tim

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


def RULmodel_SN_5(input_shape):
	#Create a sequential model
	model = Sequential()
	
	#Add the layers for the model
	model.add(Dense(20, input_dim=input_shape, activation='relu', kernel_initializer='glorot_normal', 
					kernel_regularizer=regularizers.L1L2(l1_lambda_regularization, l2_lambda_regularization), 
					name='fc1'))
	model.add(Dense(20, input_dim=input_shape, activation='relu', kernel_initializer='glorot_normal', 
					kernel_regularizer=regularizers.L1L2(l1_lambda_regularization, l2_lambda_regularization), 
					name='fc2'))
	model.add(Dense(1, activation='linear', name='out'))
	
	return model


def get_compiled_model(model_def, shape, model_type='ann'):

	global global_counter

	if global_counter == 10:
		print("clearing session")
		K.clear_session()  #Clear the previous tensorflow graph
		global_counter = 0
	#tf.reset_default_graph()
	
	#Shared parameters for the models
	optimizer = Adam(lr=0, beta_1=0.5)
	lossFunction = "mean_squared_error"
	metrics = ["mse"]
	model = None

	#Create and compile the models

	if model_type=='ann':
		model = model_def(shape)
		model.compile(optimizer = optimizer, loss = lossFunction, metrics = metrics)
	else:
		pass

	return model


def new_function():

	models = {'shallow-20':RULmodel_SN_5}
	shape = 20

	for i in range(100):
		test = get_compiled_model(models['shallow-20'], shape, model_type='ann')
		test.summary()

def run_exhaustive_search():

	global global_counter
	#Perform exhaustive search to find the optimal parameters

	models = {'shallow-20':RULmodel_SN_5}

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

	model = get_compiled_model(models['shallow-20'], shape, model_type='ann')
	tunable_model = SequenceTunableModelRegression('ModelRUL_SN_5', model, lib_type='keras', data_handler=dHandler_cmaps,
												  epochs=20)

	min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
	tunable_model.data_scaler = min_max_scaler

	for dataset_number in max_window_size:
		
		print("Results for dataset "+dataset_number)
		file = open("results/MLP/exhauxtive_search_"+dataset_number+".csv", "w")
		start_time = tim.clock()
		
		tunable_model.data_handler.change_dataset(dataset_number)
		
		writer = csv.writer(file)
		verbose = 1
		
		for r in range(90, 141):   #Load max_rul first as it forces reloading the dataset from file
			
			verbose = 1
			tunable_model.data_handler.max_rul = r
			
			for w in range(15, max_window_size[dataset_number]+1):
			
				for s in range(1,11):
					
					print("Testing for w:{}, s:{}, r:{}".format(w, s, r))
					
					#Set data parameters
					tunable_model.data_handler.sequence_length = w
					tunable_model.data_handler.sequence_stride = s

					#Create and compile the models
					shape = num_features*w
					model = get_compiled_model(models['shallow-20'], shape, model_type='ann')

					#Add model to tunable model
					tunable_model.change_model('ModelRUL_SN', model, 'keras')
									
					#Load the data
					tunable_model.load_data(unroll=True, verbose=verbose, cross_validation_ratio=0)
					
					#Train and evaluate
					tunable_model.train_model(learningRate_scheduler=lrate, verbose=0)
					tunable_model.evaluate_model(['rhs', 'rmse'], round=2)


					cScores = tunable_model.scores
					rmse = math.sqrt(cScores['score_1'])
					rmse2 = cScores['rmse']
					rhs = cScores['rhs']
					time = tunable_model.train_time
					
					row = [w, s, r, rmse, rhs]
					writer.writerow(row)
					
					#msgStr = "The model variables are " + str(x) + "\tThe scores are: [RMSE:{:.4f}, RHS:{:.4f}]\n".format(rmse, rhs)
					#file.write(msgStr)

					global_counter = global_counter + 1
					
		end_time = tim.clock()
		file.close()
		totalTime[dataset_number] = end_time - start_time


run_exhaustive_search()
#new_function()
