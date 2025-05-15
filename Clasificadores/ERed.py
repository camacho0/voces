from keras.models import Sequential
from keras.layers import *
from pandas.io.stata import stata_epoch
from sklearn.model_selection import train_test_split
#import numpy as np
import pandas as pd

data= pd.read_csv("Dataset_vocesp.csv")
#data= data.loc[:340,:]

Y=data.Clase
X=data.drop(["Clase"],axis=1)

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)

print(X_train.shape,Y_train.shape)
# Crear modelo secuencial
model = Sequential()

# Añadir capa oculta con 64 neuronas, activación relu
model.add(Dense(10, input_dim=45, activation='sigmoid'))
#model.add(Dropout(0.05))

# Añadir otra capa oculta con 32 neuronas, activación relu
model.add(Dense(5, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(248,activation='sigmoid'))
#model.add(Dropout(0.25))
#model.add(Dense(32, activation='sigmoid'))
# Añadir capa de salida con 1 neurona, activación sigmoidal
model.add(Dense(1, activation='sigmoid'))

# Compilar modelo con función de pérdida binary_crossentropy y optimizador adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=500)
