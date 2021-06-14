from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

class MyFlatten(Layer):
    def __init__(self, data_format=None, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        return K.batch_flatten(inputs)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))