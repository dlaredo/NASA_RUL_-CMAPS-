import os
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sklearn

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
    
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    
    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        print("Logs:", logs.items())
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        print("Val_logs:", val_logs)
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        
        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

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