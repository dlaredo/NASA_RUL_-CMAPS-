import numpy as np
import random
import pandas as pd

from sequenced_data_handler import SequenceDataHandler

class CMAPSDataHandler(SequenceDataHandler):

	'''
    5    T2        - Total temperature at fan inlet      R
    6    T24       - Total temperature at lpc outlet     R
    7    T30       - Total temperature at hpc outlet     R
    8    T50       - Total temperature at LPT outlet     R
    9    P2        - Pressure at fan inlet               psia
    10   P15       - Total pressure in bypass-duct       psia
    11   P30       - Total pressure at HPC outlet        psia
    12   Nf        - Physical fan speed                  rpm
    13   Nc        - Physical core speed                 rpm
    14   epr       - Engine Pressure ratio (P50/P2)      --
    15   Ps30      - Static Pressure at HPC outlet       psia
    16   phi       - Ratio fuel flow to Ps30             pps/psi
    17   NRf       - corrected fan speed                 rpm
    18   NRc       - Corrected core speed                rpm
    19   BPR       - Bypass ratio                        --
    20   farB      - Burner fuel-air ratio               --
    21   htBleed   - Bleed enthalpy                      --
    22   Nf_dmd    - Demanded fan speed                  rpm
    23   PCNfR_dmd - Demanded corrected fan speed        rpm
    24   W31       - HPT coolant bleed                   lbm/s
    25   W32       - LPT coolant bleed                   lbm/s
    '''


	#Method definition

	def __init__(self, data_folder, dataset_number, selected_features, max_rul, sequence_length=1, sequence_stride=1, data_scaler=None):

		#Public properties
		self._selected_features = selected_features
		self._max_rul = max_rul
		self._rectify_labels = False


		#ReadOnly properties
		self._dataset_number = str(dataset_number)
		self._data_folder = data_folder
		self._file_train_data = data_folder+'/train_FD00'+self.dataset_number+'.txt'
		self._file_test_data = data_folder+'/test_FD00'+self.dataset_number+'.txt'
		self._file_rul = data_folder+'/RUL_FD00'+self.dataset_number+'.txt'
		#self._load_from_file = True
		
		self._column_names = {0:'Unit Number', 1:'Cycle', 2:'Op. Settings 1', 3:'Op. Settings 2', 4:'Op. Settings 3', 5:'T2',
			6:'T24', 7:'T30', 8:'T50', 9:'P2', 10:'P15', 11:'P30', 12:'Nf', 13:'Nc', 14:'epr', 15:'Ps30', 
			16:'phi', 17:'NRf', 18:'NRc', 19:'BPR', 20:'farB', 21:'htBleed', 22:'Nf_dmd', 23:'PCNfR_dmd', 
			24:'W31', 25:'W32'}

		self._df_train = None
		self._df_test = None

		print("init")

		#super init
		super().__init__(sequence_length=sequence_length, sequence_stride=sequence_stride, feature_size=len(selected_features), data_scaler=data_scaler)


	def load_csv_into_df(self, file_name):
		"""Given the filename, load the data into a pandas dataframe"""

		df = pd.read_csv(file_name ,sep='\s+',header=None)
		df.rename(columns = self._column_names, inplace=True)

		return df


	def generate_df_with_rul(self, df):
		"""Given a dataframe compute its RUL and extract its selectedFeatures"""

		gruoped_by_unit = df.groupby('Unit Number')
		rul_vector = gruoped_by_unit.size().values
		num_units = len(gruoped_by_unit)

		#print("from aux functions")
		#print(num_units)

		if self._max_rul > 0:
			trimmed_rul = rul_vector - self._max_rul

		#print(trimmed_rul)

		df['RUL'] = df.apply(compute_training_rul, axis = 1, args = (rul_vector, self._max_rul, None))
		selected_features_rul = self._selected_features[:]
		selected_features_rul.extend(['Unit Number', 'RUL'])
		df_selected_features = df[selected_features_rul]
		
		return df_selected_features, num_units, trimmed_rul


	def change_dataset(self, dataset_number, data_folder=None):
		"""Change the current dataset"""

		self._dataset_number = str(dataset_number)

		if data_folder == None:
			data_folder = self._data_folder
		else:
			self._data_folder = data_folder

		self._file_train_data = data_folder+'/train_FD00'+self._dataset_number+'.txt'
		self._file_test_data = data_folder+'/test_FD00'+self._dataset_number+'.txt'
		self._file_rul = data_folder+'/RUL_FD00'+self._dataset_number+'.txt'
		
		self._load_data_from_origin = True


	def create_dataFrames(self, verbose=0, cross_validation_ratio=0):
		"""Load the data from the files and create the corresponding dataframes"""

		if verbose == 1:
			print("Loading data for dataset {} with window_size of {}, stride of {} and maxRUL of {}. Cros-Validation ratio {}".format(self._dataset_number, 
				self._sequence_length, self._sequence_stride, self._max_rul, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		self._df_train = self.load_csv_into_df(self._file_train_data)
		self._df_test = self.load_csv_into_df(self._file_test_data)
		self._df_train, num_units, trimmed_rul_train = self.generate_df_with_rul(self._df_train)

		#Reescale data if data_scaler is available
		if self._data_scaler != None:

			df_cols = self._df_train.columns
			#print(df_cols)
			cols_normalize = self._df_train.columns.difference(['Unit Number', 'RUL'])
			#print(cols_normalize)

			#Reescale training data
			norm_train_df = pd.DataFrame(self._data_scaler.fit_transform(self._df_train[cols_normalize]), columns=cols_normalize, index=self._df_train.index)
			join_df = self._df_train[self._df_train.columns.difference(cols_normalize)].join(norm_train_df)
			self._df_train = join_df.reindex(columns = df_cols)
			#print(self._df_train.head())

			#Rescale test data
			norm_test_df = pd.DataFrame(self._data_scaler.transform(self._df_test[cols_normalize]), columns=cols_normalize, index=self._df_test.index)
			join_df = self._df_test[self._df_test.columns.difference(cols_normalize)].join(norm_test_df)
			self._df_test = join_df.reindex(columns = df_cols)
			#print(self._df_test.head())


	def create_lists(self, cross_validation_ratio=0):
		"""From the dataframes create the lists containing the necessary data containing the samples"""

		#Modify properties in the parent class, and let the parent class finish the data processing
		self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list = self.generate_train_arrays(cross_validation_ratio)
		self._X_test_list = self.generate_test_arrays()
		self._y_test_list = np.loadtxt(self._file_rul)


	def load_data(self, unroll=True, cross_validation_ratio=0, verbose=0):
		"""Load the data using the specified parameters"""

		if self._load_data_from_origin == True:
			print("Loading data from file")
			self.create_dataFrames(verbose = verbose, cross_validation_ratio = cross_validation_ratio)
		else:
			print("Loading data from memory")

		self.create_lists(cross_validation_ratio)

		self.generate_train_data(unroll)
		self.generate_test_data(unroll)

		self._load_data_from_origin = False #As long as the dataframe doesnt change, there is no need to reload from file


	def generate_train_arrays(self, cross_validation_ratio=0):
		"""From the dataframes generate the feature arrays and their labels"""

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()

		if cross_validation_ratio < 0 or cross_validation_ratio > 1 :
			print("Error, cross validation must be between 0 and 1")
			return

		df_train = self._df_train

		if cross_validation_ratio != 0:

			df_train, df_crossVal, num_train, num_crossVal = self.split_dataFrames(df_train, cross_validation_ratio)
			df_crossVal, rul_crossVal = self.generate_cross_validation_from_df(df_crossVal, self._sequence_length)

			#Create a list with the data from cv dataframe
			grouped_by_unit = df_crossVal.groupby('Unit Number')
			for engine_number, df in grouped_by_unit:
				data = df[self._selected_features].values
				X_crossVal_list.append(data)
				y_crossVal_list = rul_crossVal


		#Create a list with the data from train dataframe
		grouped_by_unit = df_train.groupby('Unit Number')
		for engine_number, df in grouped_by_unit:
			data = df[self._selected_features].values
			labels = df['RUL'].values
			X_train_list.append(data)
			y_train_list.append(labels)

		#X, y = get_X_y_from_df(df_rul, time_window, selected_features, num_units, dataset_type, stride=stride)

		return X_train_list, y_train_list, X_crossVal_list, y_crossVal_list


	def generate_test_arrays(self):
		"""From the dataframes generate the feature arrays and their labels"""

		X_test_list = list()

		#Create a list with the data from test dataframe
		grouped_by_unit = self._df_test.groupby('Unit Number')

		for engine_number, df in grouped_by_unit:
			data = df[self._selected_features].values
			X_test_list.append(data)

		return X_test_list


	def split_dataFrames(self, df, splitting_ratio):
		"""Split the dataframes according to the indicated splitting ratio"""

		num_engines = df['Unit Number'].max()

		shuffled_engines = list(range(1,num_engines+1))
		random.shuffle(shuffled_engines)

		i = int(splitting_ratio*num_engines)
		num_crossVal = i
		num_train = num_engines - num_crossVal

		crossVal_engines = shuffled_engines[:i]
		train_engines = shuffled_engines[i:]

		df_train = df[df['Unit Number'].isin(train_engines)]
		df_crossVal = df[df['Unit Number'].isin(crossVal_engines)]

		return (df_train, df_crossVal, num_train, num_crossVal)

	
	"""
	def split_dataFrames(self, df, trimmed_rul, splitting_ratio):

		num_engines = df['Unit Number'].max()

		shuffled_engines = list(range(1,num_engines+1))
		random.shuffle(shuffled_engines)

		i = int(splitting_ratio*num_engines)
		num_crossVal = i
		num_train = num_engines - num_crossVal

		crossVal_engines = shuffled_engines[:i]
		train_engines = shuffled_engines[i:]
		trimmed_rul_train = trimmed_rul[:i]
		trimmed_rul_crossVal = trimmed_rul[i:]

		df_train = df[df['Unit Number'].isin(train_engines)]
		df_crossVal = df[df['Unit Number'].isin(crossVal_engines)]

		return (df_train, df_crossVal, num_train, num_crossVal, trimmed_rul_train, trimmed_rul_crossVal)
	"""


	def generate_cross_validation_from_df(self, df, sequence_length):
		"""Given a dataframe truncate the data to generate cross validation dataset"""
		


		"""Have to fix this in case I want sequence length larger than real size"""
		




		data = []
		
		grouped_by_unit = df.groupby('Unit Number')
		sizes = grouped_by_unit.size().values
		ruls = np.zeros((sizes.shape[0], ))
		cols = df.columns
		
		count = 0
		
		#Truncate readings up to a random number larger than window size but less than total size
		for engine_number, df in grouped_by_unit:
			truncateAt = random.randint(sequence_length, sizes[count])
			ruls[count] = sizes[count] - truncateAt
			data_temp = df.values[:truncateAt]

			if count == 0:
				data = data_temp
			else:
				data = np.concatenate([data, data_temp])
			
			count = count + 1
		
		df = pd.DataFrame(data=data, columns=cols)
		
		return df, ruls


	#Property definition

	@property
	def selectedFeatures(self):
		return self._selectedFeatures

	@selectedFeatures.setter
	def selectedFeatures(self, selectedFeatures):
		self._selectedFeatures = selectedFeatures

	@property
	def max_rul(self):
		return self._max_rul

	@max_rul.setter
	def max_rul(self, max_rul):
		self._max_rul = max_rul

	@property
	def rectify_labels(self):
		return self._rectify_labels

	@rectify_labels.setter
	def rectify_labels(self, rectify_labels):
		self._rectify_labels = rectify_labels

	#ReadOnly Properties
	
	@property
	def dataset_number(self):
		return self._dataset_number

	@property
	def data_folder(self):
		return self._data_folder

	@property
	def file_train_data(self):
		return self._file_train_data

	@property
	def file_test_data(self):
		return self._file_test_data

	@property
	def file_rul(self):
		return self._file_rul

	"""	
	@property
	def load_from_file(self):
		return self._load_from_file
	"""

	@property
	def column_names(self):
		return self._column_names

	@property
	def df_train(self):
		return self._df_train

	@property
	def df_test(self):
		return self._df_test



#Auxiliary functions

def compute_training_rul(df_row, *args):
	"""Compute the RUL at each entry of the DF"""

	max_rul = args[1]
	rul_vector = args[0]
	rul_vector_index = int(df_row['Unit Number']) - 1


	if max_rul > 0 and rul_vector[rul_vector_index] - df_row['Cycle'] > max_rul:
		return max_rul
	else:
		return rul_vector[rul_vector_index] - df_row['Cycle']




