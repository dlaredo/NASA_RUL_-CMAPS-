from __future__ import print_function
import os, math, random, pickle, time
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler,EarlyStopping
from keras.models import Sequential, Model, load_model
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation, Input, merge, Convolution2D, Reshape, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
import keras
from sklearn import preprocessing
from keras import backend as K
from keras import metrics
import matplotlib.pyplot as plt

def get_score(y_true, y_pred):
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

FeatureN = 14
nb_epoch = 250
batch_size = 512
FilterN = 10
FilterL = 10
rmse,sco,tm = [], [], []


#writer = open('DCNN_5C_30TW_noRearly.pkl', 'wb')
ConstRUL = 125
TW = 30
Dataset = '2'
# Time Window Max of Dataset
#1:31, 2:21, 3:38, 4:19
#1:125, 2: 155, 3:125， 4: 155

############ training samples ##################################

setTrain = {'1':100, '2':260, '3':100, '4':248}
setTest = {'1':100, '2':259, '3':100, '4':248}
nTrain = setTrain[Dataset]
nTest = setTest[Dataset]

data = [] 
for line in open("DAT/train_FD00"+Dataset+".txt"):
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
        label[i].append(ConstRUL if length-j-1>=ConstRUL else length-j-1)
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
for line in open("DAT/test_FD00"+Dataset+".txt"):
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
for line in open("DAT/RUL_FD00"+Dataset+".txt"):
    labelt.append(line.split())
labelt = np.cast['int32'](labelt)
labelnew = []
for i in range(nTest):
    if i not in count_miss:
        #labelnew.append(labelt[i][0])
        labelnew.append(labelt[i][0] if labelt[i][0]<=ConstRUL else ConstRUL)
labelt = labelnew
labelt=np.array(labelt)

seed = 2222
np.random.seed(seed)
np.random.shuffle(samples)
np.random.seed(seed)
np.random.shuffle(targets)
samplet = samplet[np.argsort(labelt)]
labelt = labelt[np.argsort(labelt)]

###########################################################

for i in range(1):
    print(i)
    start = time.clock()
    input_layer = Input(shape=(TW, FeatureN))
    y = Reshape((TW, FeatureN, 1), input_shape=(TW, FeatureN, ),name = 'Reshape')(input_layer)

    y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C1')(y)
    y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C2')(y)
    y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C3')(y)
    y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C4')(y)
    #y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C5')(y)
    #y = Convolution2D(FilterN, FilterL, 1, border_mode='same', init='glorot_normal', activation='tanh', name='C6')(y)
    
    y = Convolution2D(1, 3, 1, border_mode='same', init='glorot_normal', activation='tanh', name='Clast')(y)  
    
    y = keras.layers.Reshape((TW,14))(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    
    #y = Dense(100, activation='tanh', init='glorot_normal', activity_regularizer=keras.regularizers.l2(0.01),)(y)
    y = Dense(100,activation='tanh', init='glorot_normal', name='fc')(y)
    y = Dense(1)(y)
    
    opt = keras.optimizers.Adam(lr=0, beta_1=0.5)
    DCNN = Model([input_layer], [y])
    #DCNN.compile(loss=get_score,optimizer=opt)
    DCNN.compile(loss='mean_squared_error',optimizer=opt)
    lrate = LearningRateScheduler(step_decay)
    history = DCNN.fit(samples, targets,nb_epoch=nb_epoch, batch_size=batch_size,verbose=1, 
                     validation_data=(samplet, labelt), callbacks=[lrate])
    
    #, TensorBoard(log_dir='tmp\\tan_4c_4')
    
    #history = DCNN.fit(samples, targets,nb_epoch=nb_epoch, batch_size=batch_size,verbose=1, 
    #                 validation_data=(samplet, labelt), callbacks=[lrate])
    #history = DCNN.fit(samples, targets,nb_epoch=nb_epoch, batch_size=batch_size,verbose=0, callbacks=[lrate])
    
    score = DCNN.evaluate(samplet, labelt, batch_size=batch_size, verbose=1)
    print('Test score:', score)
    end = time.clock()
    
    rmse.append(np.sqrt(score))
    sco.append(get_score(labelt, DCNN.predict(samplet)))
    tm.append(end-start)
    
    #DCNN.save('DCNN_5C_noR_'+str(i)+'.h5')
    #DCNN.save_weights('DCNN_5C_'+str(i))

rec = np.array([[np.average(rmse), np.std(rmse)],[np.average(sco), np.std(sco)],[np.average(tm), np.std(tm)]])
print(rec)
pickle.dump(rec, writer)

#np.savetxt('file.txt', rec, delimiter=' ')
#DCNN.save_weights('DCNN_5C')

writer.close()

#writer = open('Dataset1_30TW_data.pkl', 'wb')
#pickle.dump([samplet, labelt], writer)
#writer.close()



