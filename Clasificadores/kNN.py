import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")
Y_train=train.Clase
X_train=train.drop(["Clase"],axis=1)
X_train=X_train.sample(frac=1,axis=0)
Y_test=test.Clase
X_test=test.drop(["Clase"],axis=1)

print("Datos")

model= neighbors.KNeighborsClassifier(15)

metri=["f1","accuracy","recall","precision"]
score_knn=cross_validate(model,X_train,Y_train,scoring=metri,cv=5,return_estimator=True)

print("Cross validation scores")
print("F1")
print(np.mean(score_knn["test_f1"]))
print(np.std(score_knn["test_f1"]))
print("Accuracy")
print(np.mean(score_knn["test_accuracy"]))
print(np.std(score_knn["test_accuracy"]))
print("Recall")
print(np.mean(score_knn["test_recall"]))
print(np.std(score_knn["test_recall"]))
#metricas
acc=[]
f1=[]
re=[]
mc=[]
for i in range(len(score_knn["estimator"])):
    Y_pred=score_knn["estimator"][i].predict(X_test)
    acc.append(metrics.accuracy_score(Y_test,Y_pred))
    f1.append(metrics.f1_score(Y_test,Y_pred))
    re.append(metrics.recall_score(Y_test,Y_pred))
    mc.append(metrics.confusion_matrix(Y_test,Y_pred))

acc=np.array(acc)
f1=np.array(f1)
re=np.array(re)
print("Test accuracy")
print(np.mean(acc),np.std(acc))
print("Test F1")
print(np.mean(f1),np.std(f1))
print("Test recall")
print(np.mean(re),np.std(re))

#modelo sin cross validation
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))
