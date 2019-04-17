import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing

def get_X_y_from_df(df, time_window, features, num_units, dataset_type, stride=1):
    """From a dataset with RUL values, create the X and y arrays using the specified time windows"""
    
    n_m = df.shape[0]
    n_x = len(features)
    
    df_values = df[features].values
    targets = df['RUL'].values
    n_m = 0
    n_X = len(features)
    df_unit_values = []
    targets_unit = []
    num_samples_unit = []
    engine_windows = []
    
    engineNumbers = df['Unit Number'].unique()
    
    #Count number of elements at each group so that we can create the matrix to hold them all. 
    #Also store each matrix in temporary arrays to access them faster
    for j in range(num_units):
        
        i = engineNumbers[j]
        df_unit = df.loc[df['Unit Number'] == i]
        df_unit_values.append(df_unit[features].values) #is this a view or a copy of the df?
        targets_unit.append(df_unit['RUL'].values) #is this a view or a copy of the df?
        num_samples_unit.append(df_unit.shape[0])
        engine_windows.append(math.floor((df_unit.shape[0]-time_window)/stride) + 1)
        n_m = n_m + engine_windows[-1]
    
    #Create the numpy arrays to hold the features
    if (dataset_type == 'train' or dataset_type == 'cross_validation'):
        X, y = np.empty([n_m, n_x*time_window]), np.empty([n_m, 1])
    else:
        X, y = np.empty([num_units, n_x*time_window]), np.empty([num_units, 1])
        
    k = 0
    
    #Create the feature matrix by moving the time window for each type of engine.
    for i in range(num_units):
    
        if (dataset_type == 'train' or dataset_type == 'cross_validation'):
            for j in range(engine_windows[i]):

                time_window_samples = df_unit_values[i][j*stride:j*stride+time_window,:]
                X[k,:] = np.squeeze(time_window_samples.reshape(1,-1))
                y[k] = targets_unit[i][j*stride+time_window-1]
                k = k + 1
        else:
            time_window_samples = df_unit_values[i][-time_window:,:]
            X[k,:] = np.squeeze(time_window_samples.reshape(1,-1))
            k = k + 1
    
    return X, y


def create_windowed_data(df, selected_features, dataset_type, time_window=10, constRUL=125, stride=1, crossValidationRatio=0):
    """Given the dataframe, reshape the data to create the time windows"""

    X_crossVal, y_crossVal = None, None

    if crossValidationRatio < 0 or crossValidationRatio > 1 :
        print("Error, cross validation must be between 0 and 1")
        return

    df_rul, num_units, trimmedRUL_train = generate_df_withRUL(df, selected_features, constRUL)

    #Split for cross-validation
    if crossValidationRatio != 0 and dataset_type == 'train': 
        df_train, df_crossVal, num_train, num_crossVal, trimmedRUL_train, trimmedRUL_crossVal = split_dataFrames(df_rul, trimmedRUL_train, crossValidationRatio)
        
        df_crossVal, rul_crossVal = generate_cross_validation_from_df(df_crossVal, time_window)
        
        X, y = get_X_y_from_df(df_train, time_window, selected_features, num_train, 
                               dataset_type, stride=stride)
        
        X_crossVal, _ = get_X_y_from_df(df_crossVal, time_window, selected_features, num_crossVal, 
                                        'test', stride=stride)
        
        y_crossVal = rul_crossVal
    else:
        X, y = get_X_y_from_df(df_rul, time_window, selected_features, num_units, dataset_type, stride=stride)
    
    return X, y, X_crossVal, y_crossVal, trimmedRUL_train


def generate_df_withRUL(df, selected_features, constRUL):
    """Given a dataframe compute its RUL and extract its selectedFeatures"""

    gruoped_by_unit = df.groupby('Unit Number')
    rul_vector = gruoped_by_unit.size().values
    num_units = len(gruoped_by_unit)

    #print("from aux functions")
    #print(num_units)

    if constRUL > 0:
        trimmedRUL = rul_vector - constRUL

    #print(trimmedRUL)

    df['RUL'] = df.apply(compute_training_rul, axis = 1, args=(rul_vector,constRUL,))
    selected_features_rul = selected_features[:]
    selected_features_rul.extend(['Unit Number', 'RUL'])
    df_selected_features = df[selected_features_rul]
    
    return df_selected_features, num_units, trimmedRUL


def split_dataFrames(df, trimmedRUL, splittingRatio):
    """Split the dataframes according to the indicated splitting ratio"""

    num_engines = df['Unit Number'].max()

    shuffledEngines = list(range(1,num_engines+1))
    random.shuffle(shuffledEngines)

    i = int(splittingRatio*num_engines)
    num_crossVal = i
    num_train = num_engines - num_crossVal

    crossVal_engines = shuffledEngines[:i]
    train_engines = shuffledEngines[i:]
    trimmedRUL_train = trimmedRUL[:i]
    trimmedRUL_crossVal = trimmedRUL[i:]

    df_train = df[df['Unit Number'].isin(train_engines)]
    df_crossVal = df[df['Unit Number'].isin(crossVal_engines)]

    return (df_train, df_crossVal, num_train, num_crossVal, trimmedRUL_train, trimmedRUL_crossVal)


def generate_cross_validation_from_df(df, window_size):
    """Given a dataframe truncate the data to generate cross validation dataset"""
    
    data = []
    
    groupedByUnit = df.groupby('Unit Number')
    sizes = groupedByUnit.size().values
    ruls = np.zeros((sizes.shape[0],1))
    cols = df.columns
    
    count = 0
    
    #Truncate readings up to a random number larger than window size but less than total size
    for engineNumber, df in groupedByUnit:
        truncateAt = random.randint(window_size, sizes[count])
        ruls[count] = sizes[count] - truncateAt
       # print("{} {} {}".format(engineNumber, truncateAt, ruls[count]))
        data_temp = df.values[:truncateAt]

        if count == 0:
            data = data_temp
        else:
            data = np.concatenate([data, data_temp])
        
        count = count + 1
    
    df = pd.DataFrame(data=data, columns=cols)
    
    return df, ruls


def compute_training_rul(df_row, *args):
    """Compute the RUL at each entry of the DF"""

    max_rul = args[1]
    rul_vector = args[0]
    rul_vector_index = int(df_row['Unit Number']) - 1

    if max_rul > 0 and rul_vector[rul_vector_index] - df_row['Cycle'] > max_rul:
        return max_rul
    else:
        return rul_vector[rul_vector_index] - df_row['Cycle']
