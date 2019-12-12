import tensorflow as tf
import helper
from tensorflow import keras

class StockModel(keras.Model):
    def __init__(self):
        super(StockModel, self).__init__()
        self.model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(7,1)),
                keras.layers.Conv1D(
                    kernel_size = 2, filters = 128, strides = 1, use_bias = True, activation = 'relu', kernel_initializer = 'VarianceScaling'
                ),
                keras.layers.AveragePooling1D(pool_size = 2, strides = 1),
                keras.layers.Conv1D(
                    kernel_size = 2, filters = 64, strides = 1, use_bias = True, activation = 'relu', kernel_initializer = 'VarianceScaling'
                ),
                keras.layers.AveragePooling1D(pool_size = 2, strides = 1),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    1, 
                    kernel_initializer = 'VarianceScaling',
                    activation = 'linear'
                )
            ]
        )
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')

if __name__ == '__main__':
    data = helper.generate('appl.csv')
    print (data)
    model = StockModel()