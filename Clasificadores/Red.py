from keras.models import Sequential
from keras.layers import *
from pandas.io.stata import stata_epoch
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")
Y_train=train.Clase
X_train=train.drop(["Clase"],axis=1)
X_train=X_train.sample(frac=1,axis=0)
Y_test=test.Clase
X_test=test.drop(["Clase"],axis=1)

# Crear modelo secuencial
model = Sequential()

# Añadir capa oculta con 64 neuronas, activación relu
model.add(Dense(10000, input_dim=45, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
#model.add(Dense(5, activation='relu'))
#model.add(Dropout(0.25))
# Añadir capa de salida con 1 neurona, activación sigmoidal
model.add(Dense(1, activation='sigmoid'))

# Compilar modelo con función de pérdida binary_crossentropy y optimizador adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=10,batch_size=1)

a,acc=model.evaluate(X_test,Y_test)
#Y_pred=model.predict(X_test)
print("Accracy:",acc)

#print(Y_pred)
#print("accuracy:" , np.sum(Y_pred == Y_test) / float(len(Y_test)))
#print("Accracy:",metrics.accuracy_score(Y_test,Y_pred))

