#Importing Libraries...

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D
from tensorflow.keras.activations import sigmoid
from scipy.io import wavfile
from tensorflow.keras.layers import Layers as Layer

mgc_dim = 180
lf0_dim = 3
vuv_dim = 1
bap_dim = 3

duration_linguistic_dim = 416
acoustic_linguisic_dim = 425
duration_dim = 5
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim

test_size = 0.112
random_state = 1234

fs = 16000
frame_period = 5
hop_length = 80
fftlen = 1024
alpha = 0.41

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]