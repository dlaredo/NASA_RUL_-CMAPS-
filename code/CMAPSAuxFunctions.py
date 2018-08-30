import matplotlib.pyplot as plt
import math
import sklearn

def step_decay(epoch):
    lrat = 0
    if epoch<100:
        lrat = 0.001
    else:
        lrat = 0.0001
    return lrat 

def plotRUL(cycles, rulArray, pred, engineUnit, methodName):
    """plot the trend of the predictions made by the predictor"""
    
    plt.clf()
    plt.plot(cycles, rulArray, 'bo-', label='RUL')
    plt.plot(cycles, nnPred, 'go-', label=methodName)
    plt.legend()
    plt.xlabel("Time (Cycle)")
    plt.ylabel("RUL")
    plt.title("Test Engine Unit #{}".format(engineUnit))
    plt.show()


def get_minibatches(X_full, y_full, batch_size):
    """Function to get minibatches from the full batch"""

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