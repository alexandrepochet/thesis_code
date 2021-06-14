import numpy as np
from keras.callbacks import Callback 


class CustomEarlyStopping(Callback):
    def __init__(self, ratio=1.3, max_accuracy_training=0.65,
                 patience=3, verbose=0):
        super(CustomEarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.less
        self.max_accuracy_training = max_accuracy_training

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_accuracy')
        current_train = logs.get('accuracy')
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if (self.monitor_op(np.divide(current_train,current_val),self.ratio) and self.monitor_op(current_train,self.max_accuracy_training)) or current_val<0.5:
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))