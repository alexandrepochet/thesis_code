import random as rn
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import time
import os
from keras import backend as K

 
def reset_seeds():

   rn.seed(0)
   np.random.seed(0) 
   #tf.random.set_random_seed(0)
   tf.compat.v1.random.set_random_seed(0)
   config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
 #  tf.set_random_seed(0)
   sess = tf.Session(graph=tf.get_default_graph(), config=config)
   K.set_session(sess)
   #sess = tf.keras.backend.get_session()
   tf.keras.backend.clear_session()
  # tf.set_random_seed(2)
   sess.close()
   del sess

# fit MLP to dataset and print error
def fit_model(X, y):
	# design network
	reset_seeds()
	model = Sequential()
	model.add(Dense(10, input_dim=1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='RMSprop')
	# fit network
	model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
	# forecast
	yhat = model.predict(X, verbose=0)
	print(mean_squared_error(y, yhat[:,0]))
 
# create sequence
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
start = time.time()
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:,0], values[:,1]
# repeat experiment
repeats = 50
for _ in range(repeats):
	fit_model(X, y)
end = time.time()
print(end - start)