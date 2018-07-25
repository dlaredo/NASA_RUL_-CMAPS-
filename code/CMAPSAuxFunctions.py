import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing

def step_decay(epoch):
    lrat = 0
    if epoch<100:
        lrat = 0.001
    else:
        lrat = 0.0001
    return lrat 


def compute_training_RUL(df_row, *args):
    
    constRUL = args[1]
    rul_vector = args[0]
    rul_vector_index = int(df_row['Unit Number']) - 1
    
    if constRUL > 0 and rul_vector[rul_vector_index] - df_row['Cycle'] > constRUL:
        return constRUL
    else:
        return rul_vector[rul_vector_index] - df_row['Cycle']


def load_into_df(from_file):
    """Given the filename, load the data into a pandas dataframe"""

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

    df = pd.read_csv(from_file ,sep='\s+',header=None)

    col_names = {0:'Unit Number', 1:'Cycle', 2:'Op. Settings 1', 3:'Op. Settings 2', 4:'Op. Settings 3', 5:'T2',
                6:'T24', 7:'T30', 8:'T50', 9:'P2', 10:'P15', 11:'P30', 12:'Nf', 13:'Nc', 14:'epr', 15:'Ps30', 
                16:'phi', 17:'NRf', 18:'NRc', 19:'BPR', 20:'farB', 21:'htBleed', 22:'Nf_dmd', 23:'PCNfR_dmd', 
                24:'W31', 25:'W32'}

    df.rename(columns=col_names, inplace=True)

    return df


def df_to_arrays_by_unit(df, time_window, features, num_units, stride=1):
    """From a dataset with RUL values, return a list of numpy arrays for each unit"""

    #n_m = 0
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
        #n_m = n_m + engine_windows[-1]

    return df_unit_values, targets_unit, engine_windows


def create_fixed_size_sequence(unit_values_list, unit_targets_list, sequence_length, padding_top=True):
    """Given the readings for each engine, create sequences of size sequence_lenght"""

    unit_values_padded = list()
    unit_targets_padded = list()
    feature_number = unit_values_list[0].shape[1]

    for unit_values, unit_targets in zip(unit_values_list, unit_targets_list):

        if padding_top == True:
            padding_values = unit_values[0,:]
            padding_target = unit_targets[0]
        else:
            padding_values = unit_values[-1,:]
            padding_target = unit_targets[-1]

        seq_len = unit_values.shape[0]
        #print(seq_len)

        #print(sequence_length)
        #print(feature_number)
        fixed_len_unit_values = np.empty([sequence_length, feature_number])
        fixed_len_unit_targets = np.empty(sequence_length,)

        #print(unit_values.shape)
        #print(fixed_len_unit_values.shape)
        #print(unit_targets.shape)
        #print(fixed_len_unit_targets.shape)


        if seq_len < sequence_length:   #Pad elements to matrix
            padding_size = sequence_length-seq_len
            padding_matrix_values = np.tile(padding_values, (padding_size,1))
            padding_matrix_targets = np.repeat(padding_target, padding_size)

            if padding_top == True:
                #Pad elements
                fixed_len_unit_values[:padding_size] =  padding_matrix_values
                fixed_len_unit_targets[:padding_size] = padding_matrix_targets

                #Add original elements
                fixed_len_unit_values[padding_size:] = unit_values
                fixed_len_unit_targets[padding_size:] = unit_targets
            else:
                #Add original elements
                fixed_len_unit_values[:seq_len] = unit_values
                fixed_len_unit_targets[:seq_len] = unit_targets

                #Pad elements
                fixed_len_unit_values[seq_len:] =  padding_matrix_values
                fixed_len_unit_targets[seq_len:] = padding_matrix_targets
        else:   
            #Truncate elements from matrix
            fixed_len_unit_values = unit_values[-sequence_length:]
            fixed_len_unit_targets = unit_targets[-sequence_length:]

        unit_values_padded.append(fixed_len_unit_values)
        unit_targets_padded.append(fixed_len_unit_targets)

    return unit_values_padded, unit_targets_padded


def get_Sequenced_X_y_from_df(df, time_window, features, num_units, dataset_type, stride=1, sequence_length=0, strategy='augment/top'):
    """From a dataset with RUL values, create sequenced X and y arrays using the specified time windows"""

    n_m = 0
    n_x = len(features)

    padding_top = True

    df_unit_values, targets_unit, _ = df_to_arrays_by_unit(df, time_window, features, num_units, stride)

    sequenceLenghts = [seq.shape[0] for seq in targets_unit]
    strategy_split = strategy.split("/")

    if len(strategy_split) == 2:
        strategy =  strategy_split[0]
        padding = strategy_split[1]
    else:
        print("Unknowkn strategy applying avg and top padding")
        strategy =  "avg"
        padding = "top"

    #The strategy for generating fixed size sequences will be either truncating or augmenting data or a mixture of both
    if sequence_length == 0:
        if strategy == "augment":
            sequence_length = max(sequenceLenghts)
            print("Increasing sequence length to " + str(sequence_length))
        elif strategy == "reduce":
            sequence_length = min(sequenceLenghts)
            print("Reducing sequence length to " + str(sequence_length))
        elif strategy == "avg":
            sequence_length = int(sum(sequenceLenghts)/len(sequenceLenghts))
            print("Using average sequence length " + str(sequence_length))
        else:
            sequence_length = int(sum(sequenceLenghts)/len(sequenceLenghts))
            print("Unknowkn strategy, applying avg " + str(sequence_length))

    if padding == "bottom":
        padding_top = False

    unit_values_padded, unit_targets_padded = create_fixed_size_sequence(df_unit_values, targets_unit, sequence_length, padding_top = padding_top)

    print(unit_values_padded[0].shape)
    print(unit_targets_padded[0].shape)
    print(unit_values_padded[0])
    print(unit_targets_padded[0])

    num_windows = math.floor((sequence_length-time_window)/stride) + 1
        
    n_m = num_windows*len(targets_unit)
    
    #Create the numpy arrays to hold the features
    if (dataset_type == 'train' or dataset_type == 'cross_validation'):
        X, y = np.empty([n_m, n_x*time_window]), np.empty([n_m, 1])
    else:
        X, y = np.empty([num_units, n_x*time_window]), np.empty([num_units, 1])
        
    k = 0
    
    #Create the feature matrix by moving the time window for each type of engine.
    for i in range(num_units):
    
        if (dataset_type == 'train' or dataset_type == 'cross_validation'):
            for j in range(num_windows):

                time_window_samples = unit_values_padded[i][j*stride:j*stride+time_window,:]
                X[k,:] = np.squeeze(time_window_samples.reshape(1,-1))
                y[k] = unit_targets_padded[i][j*stride+time_window-1]
                k = k + 1
        else:
            time_window_samples = unit_values_padded[i][-time_window:,:]
            X[k,:] = np.squeeze(time_window_samples.reshape(1,-1))
            k = k + 1

    return X, y


def get_X_y_from_df(df, time_window, features, num_units, dataset_type, stride=1):
    """From a list of numpy arrays for each unit, create non sequenced X and y arrays using the specified time windows"""

    n_m = 0
    n_x = len(features)

    df_unit_values, targets_unit, engine_windows = df_to_arrays_by_unit(df, time_window, features, num_units, stride)

    for num_windows in engine_windows:
        n_m = n_m + num_windows
    
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


def create_windowed_data(df, selected_features, dataset_type, time_window=10, constRUL=125, stride=1, sequenced=False, sequenceLength=0, sequence_strategy='avg/top', crossValidationRatio=0):
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
        
        if sequenced == True:
            X, y = get_Sequenced_X_y_from_df(df_train, time_window, selected_features, num_train, dataset_type, stride=stride, 
                sequence_length=sequenceLength, strategy=sequence_strategy)
            
        else:
            X, y = get_X_y_from_df(df_train, time_window, selected_features, num_train, dataset_type, stride=stride)
        
        X_crossVal, _ = get_X_y_from_df(df_crossVal, time_window, selected_features, num_crossVal, 'test', stride=stride)
        
        y_crossVal = rul_crossVal
    else:
        if sequenced == True:
            X, y = get_Sequenced_X_y_from_df(df_rul, time_window, selected_features, num_units, dataset_type, stride=stride, 
                sequence_length=sequenceLength, strategy=sequence_strategy)
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

    df['RUL'] = df.apply(compute_training_RUL, axis = 1, args=(rul_vector,constRUL,))
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


#def getSpecificEngine
 

def get_X_y_from_Dataset(Dataset, dataSetLocation, constRUL, TW):



    ############ training samples ##################################

    setTrain = {'1':100, '2':260, '3':100, '4':248}
    setTest = {'1':100, '2':259, '3':100, '4':248}
    nTrain = setTrain[Dataset]
    nTest = setTest[Dataset]

    data = [] 
    for line in open(dataSetLocation+"/train_FD00"+Dataset+".txt"):
        data.append(line.split())
    data=np.array(data)
    data = np.cast['float64'](data)
    data_copy = data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = min_max_scaler.fit_transform(data)
    num=[]
    for i in range(nTrain):
        tmp = data[np.where(data_copy[:,0]==i+1),:][0][:, np.array([6,7,8,11,12,13,15,16,17,18,19,21,24,25])]
        num.append(tmp)
    num=np.array(num)

    label=[]
    for i in range(nTrain):
        label.append([])
        length = len(num[i])
        for j in range(length):
            label[i].append(constRUL if length-j-1>=constRUL else length-j-1)
    label = np.array(label)

    samples,targets,noofsample = [],[],[]
    for i in range(nTrain):
        noofsample.append(len(num[i])-TW+1)
        for j in range(noofsample[-1]):
            samples.append(num[i][j:j+TW,:])
            targets.append(label[i][j+TW-1])
    samples = np.array(samples)
    targets = np.array(targets)

    ################## testing data ###########################
    data = [] 
    for line in open(dataSetLocation+"/test_FD00"+Dataset+".txt"):
        data.append(line.split())
    data=np.array(data)
    data = np.cast['float64'](data)
    data_copy = data
    data = min_max_scaler.transform(data)
    numt=[]
    for i in range(nTest):
        tmp = data[np.where(data_copy[:,0]==i+1),:][0][:, np.array([6,7,8,11,12,13,15,16,17,18,19,21,24,25])]
        numt.append(tmp)
    numt=np.array(numt)

    samplet, count_miss = [],[]
    for i in range(nTest):
        if len(numt[i])>=TW:
            samplet.append(numt[i][-TW:,:])
        else:
            count_miss.append(i)
    samplet = np.array(samplet)

    labelt = [] 
    for line in open(dataSetLocation+"/RUL_FD00"+Dataset+".txt"):
        labelt.append(line.split())
    labelt = np.cast['int32'](labelt)
    labelnew = []
    for i in range(nTest):
        if i not in count_miss:
            #labelnew.append(labelt[i][0])
            labelnew.append(labelt[i][0] if labelt[i][0]<=constRUL else constRUL)
    labelt = labelnew
    labelt=np.array(labelt)

    return samples, targets, samplet, labelt


def retrieve_and_reshape_data_Xiang(from_file, selected_features, time_window, dataset_type, constRUL = 125, unit_number=None, 
                              scaler=None, fit_transform=True, stride=1, crossValidationRatio = 0):
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

    
    df = pd.read_csv(from_file ,sep='\s+',header=None)

    col_names = {0:'Unit Number', 1:'Cycle', 2:'Op. Settings 1', 3:'Op. Settings 2', 4:'Op. Settings 3', 5:'T2',
                6:'T24', 7:'T30', 8:'T50', 9:'P2', 10:'P15', 11:'P30', 12:'Nf', 13:'Nc', 14:'epr', 15:'Ps30', 
                16:'phi', 17:'NRf', 18:'NRc', 19:'BPR', 20:'farB', 21:'htBleed', 22:'Nf_dmd', 23:'PCNfR_dmd', 
                24:'W31', 25:'W32'}

    #To replicate the way Xiang does his standarization
    if scaler != None:
        df_values = df.values
        engineNumbers = df_values[:,0] 
        cycleNumbers = df_values[:,1]

        if fit_transform == True:
            df_values = scaler.fit_transform(df_values)
        else:
            df_values = scaler.transform(df_values)

        df = pd.DataFrame(df_values)
        df.iloc[:, 0] = engineNumbers[:].astype(int)
        df.iloc[:, 1] = cycleNumbers[:].astype(int)
    #Up to here Xiang standarization

    df.rename(columns=col_names, inplace=True)

    #In case a specific unit number is needed
    if unit_number != None:
        df = df[df['Unit Number'] == unit_number]
        df['Unit Number'] = 1 

    gruoped_by_unit = df.groupby('Unit Number')
    rul_vector = gruoped_by_unit.size().values
    num_units = len(gruoped_by_unit)

    df['RUL'] = df.apply(compute_training_RUL, axis = 1, args=(rul_vector,constRUL,))
    selected_features_rul = selected_features[:]
    selected_features_rul.extend(['Unit Number', 'RUL'])
    df_selected_features = df[selected_features_rul]

    X, y = get_X_y_from_df(df_selected_features, time_window, selected_features, num_units, dataset_type, stride=stride)
    
    return X, y, scaler

