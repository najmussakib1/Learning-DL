import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# prottek hidden layers er por dropout layer add kora hoyeche jate overfitting komano jay. Dropout layer randomly neurons ke deactivate kore training er somoy, jate model beshi generalize korte pare.
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout layer added for regularization 
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),  # Another Dropout layer added for regularization
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])