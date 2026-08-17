import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
# print(df)
df.drop('CustomerId',axis='columns',inplace=True)
df.drop('RowNumber',axis='columns',inplace=True)
df.drop('Surname',axis='columns',inplace=True)
df['Gender'].replace({'Female':1,'Male':0},inplace=True)
main_df = pd.get_dummies(data=df, columns=['Geography']) # one hot encoding

#checked is there any null values in dataset
for columns in main_df.columns:
    print(columns)
    print(main_df[columns].isnull().sum())

print(main_df)
X = main_df.drop('Exited',axis='columns')
y = main_df['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=5)
print(X_train.shape)

model = keras.Sequential([
    keras.layers.Dense(2000,  activation='relu'),
    keras.layers.Dense(100,  activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test,y_test)
y_pred_prev = model.predict(X_test)
y_pred = (y_pred_prev >= 0.5).astype(int).flatten()
from sklearn.metrics import  classification_report

print(classification_report(y_test,y_pred))