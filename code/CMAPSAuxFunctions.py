import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing


def compute_health_score(y_true, y_pred):
    s=0
    for i in range(len(y_true)):
        d = y_pred[i] - y_true[i]
        if d < 0:
            s+=math.e**(-d/13)-1
        else:
            s+=math.e**(d/10)-1
    return s


def step_decay(epoch):
    lrat = 0
    if epoch<200:
        lrat = 0.001
    else:
        lrat = 0.0001
    return lrat 


def compute_training_RUL(df_row, *args):
    
    constRUL = args[1]
    rul_vector = args[0]
    
    if rul_vector[int(df_row['Unit Number']) - 1] - df_row['Cycle'] > constRUL:
        return constRUL
    else:
        return rul_vector[int(df_row['Unit Number']) - 1] - df_row['Cycle']


def get_X_y_from_df(df, time_window, features, num_units, dataset_type, stride=1):
    
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
    
    #Count number of elements at each group so that we can create the matrix to hold them all. 
    #Also store each matrix in temporary arrays to access them faster
    for i in range(1,num_units+1):
        
        df_unit = df.loc[df['Unit Number'] == i]
        df_unit_values.append(df_unit[features].values) #is this a view or a copy of the df?
        targets_unit.append(df_unit['RUL'].values) #is this a view or a copy of the df?
        num_samples_unit.append(df_unit.shape[0])
        engine_windows.append(math.floor((num_samples_unit[i-1]-time_window)/stride) + 1)
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
            #print(dataset_type)
            time_window_samples = df_unit_values[i][-time_window:,:]
            X[k,:] = np.squeeze(time_window_samples.reshape(1,-1))
            k = k + 1
    
    return X, y


def retrieve_and_reshape_data(from_file, selected_features, time_window, dataset_type, constRUL = 125, unit_number=None, 
                              scaler=None, fit_transform=True, stride=1):
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

