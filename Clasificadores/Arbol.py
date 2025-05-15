import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")
Y_train=train.Clase
X_train=train.drop(["Clase"],axis=1)
X_train=X_train.sample(frac=1,axis=0)
Y_test=test.Clase
X_test=test.drop(["Clase"],axis=1)


tree=DecisionTreeClassifier(criterion="gini",
                            splitter="random")

metri=["f1","accuracy","recall","precision"]
score_tree=cross_validate(tree,X_train,Y_train,scoring=metri,cv=20,return_estimator=True)
print("Cross validation scores")
print("F1")
print(np.mean(score_tree["test_f1"]))
print(np.std(score_tree["test_f1"]))
print("Accuracy")
print(np.mean(score_tree["test_accuracy"]))
print(np.std(score_tree["test_accuracy"]))
print("Recall")
print(np.mean(score_tree["test_recall"]))
print(np.std(score_tree["test_recall"]))
#metricas
#print(score_tree["estimator"][0].predict(X_test))
acc=[]
f1=[]
re=[]
mc=[]
for i in range(len(score_tree["estimator"])):
    #val.append(score_tree["estimator"][i].score(X_test,Y_test))
    Y_pred=score_tree["estimator"][i].predict(X_test)
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

#Se entrena el modelo
tree.fit(X_train,Y_train)

Y_pred=tree.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))
ma=metrics.confusion_matrix(Y_test,Y_pred)
print(ma)
