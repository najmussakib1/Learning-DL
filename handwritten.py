import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
(xtrain,ytrain), (xtest,ytest) = keras.datasets.mnist.load_data()
# print(xtrain.shape)
# plt.matshow(xtrain[0])
# plt.show()

# now flatten
xtrain_flattened = xtrain.reshape(len(xtrain), 28*28)
xtest_flattened = xtest.reshape(len(xtest), 28*28)
xtrain_flattened = xtrain_flattened / 255.0;
xtest_flattened = xtest_flattened / 255.0;
# print(xtrain_flattened[0].shape)

# neural network

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
# 10 ta level er output, 784 er input array
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(xtrain_flattened, ytrain, epochs=5)

sample = xtest_flattened[0].reshape(1, 784)

p = model.predict(sample)

print("Probabilities:", p[0])
print("Predicted digit:", np.argmax(p[0]))
print("Actual digit:", ytest[0])


# confusion matrix
# cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

# import seaborn as sn
# plt.figure(figsize = (10,7))
# sn.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')


# hidden layer
# model = keras.Sequential([
#     keras.layers.Dense(100, input_shape=(784,), activation='relu'),
#     keras.layers.Dense(10, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train_flattened, y_train, epochs=5)


# if dont. want to use flattened 
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(10, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=10)