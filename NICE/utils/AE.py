import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model
from math import ceil

def AE_model(input_dim,center_dim):
    # encoder
    input = Input(shape=(input_dim,))
    encoded = Dense(int(ceil(input_dim/2)), activation='relu')(input)
    encoded = Dense(int(ceil(input_dim/4)), activation='relu')(encoded)
    encoded = Dense(center_dim, activation='relu')(encoded)

    decoded = Dense(int(ceil(input_dim/4)), activation='relu')(encoded)
    decoded = Dense(int(ceil(input_dim/2)), activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    ae = Model(input, decoded)
    ae.compile(optimizer='adam', loss='mse')
    return ae