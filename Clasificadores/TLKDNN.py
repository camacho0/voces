import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ada import DKNN


class TLKDNN:
    def __init__(self, kb_factor=1.4, metric='heom', nom=[]):
        self.NN1st = []
        self.NN2nd = []
        self.metric = metric
        self.nom = nom
        self.kb_factor =  kb_factor

        self.dnn = DKNN()

    def getMetric(self, x1, x2):
        if self.metric == 'euclidean':
            return self.euclideanD(x1, x2)
        elif self.metric == 'manhattan':
            return self.manhattanD(x1, x2)
        elif self.metric == 'chebyshev':
            return self.chebyshevD(x1, x2)
        elif self.metric == 'heom':
            return self.heom(x1, x2)
    
    def fit(self, X, y, X_test, y_test):
        self.X_train = X
        self.X_test = X_test
        self.X_full = np.concatenate((X, X_test), axis = 0)

        self.y_train = y
        self.m1, self.n1 = X.shape # m1 vectores de entrenamiento con n caracteristicas
        self.m2, self.n2= X_test.shape # m2 vectores de prueba
        self.tags = np.unique(y) # lista de clases
        
        # rangos para la metrica heom
        if self.metric == 'heom':
            self.ranges = [0]*len(self.X_train[0])

            for i in range(0,len(self.X_train[0])):
                if not (i in self.nom):
                    self.ranges[i] = max(self.X_train[:,i])-min(self.X_train[:,i])

        self.dist_matrix() # Crea la matriz de distancias
        
        self.dnn.fit(X, y, X_test.shape[0])
        self.dnn.predict(X_test, y_test)
        #print('ADA CALCULADA')

    def dist_matrix(self):
        m = self.m1 + self.m2
        self.Dmatrix = np.zeros((m, m))
        for i in range(m-1):
            for j in range(i+1, m):
                #aux = self.heom(self.X_full[i], self.X_full[j])
                #aux = self.hvdm.hvdm(self.X_full[i], self.X_full[j])
                aux = self.heom(self.X_full[i], self.X_full[j])
                if i == j:
                    self.Dmatrix[i][j] = np.nan
                    self.Dmatrix[j][i] = np.nan
                else:
                    self.Dmatrix[i][j] = aux
                    self.Dmatrix[j][i] = aux
        

    def euclideanD(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))    

    def manhattanD(self, x1, x2):
        return np.sum(np.absolute(x1-x2))

    def chebyshevD(self, x1, x2):
        return np.amax(np.absolute(x1-x2))

    def heom(self, x1, x2):
        dis = 0
        for i in range(0,len(x1)):
            if i in self.nom:
                if x1[i] == x2[i]:
                    aux = 0
                else:
                    aux = 1
            else:
                aux = abs(x1[i]-x2[i])/self.ranges[i]
            dis += aux*aux
        return np.sqrt(dis)
    
    def _calculatekNN(self, x_i, k, data = []): # x_i es un indice
        if len(data) == 0:
            data = self.X_train

        #distances = [self.getMetric(x, x0) for x0 in data]
        distances = self.Dmatrix[x_i, 0:self.m1]

        k_i = np.argsort(distances)[:int(k)]

        return k_i

    def predict(self, X_test, weighted=False):
        predictions = []
        for i, ki in zip(range(len(X_test)), self.dnn.k_med):
            #print('ITERACION')
            self.getNN1st(self.m1+i, int(ki))
            self.getNN2nd(self.m1+i, int(ki))
            self.getNNext(self.m1+i)
            self.getNNtwo(self.m1+i, int(ki))
            if not weighted:
                y = self.votes()
            else:
                y = self.weighted_votes(i)
            predictions.append(y)
        return predictions

    def accuracy(self, preds, test):
        n = len(test)
        '''predicts = np.array(preds).reshape(1, n)
        y_test = test.reshape(1,n)

        acc = sum(predicts == y_test)/len(y_test)'''
        aux = 0
        for i, j in zip(preds, test):
            if i == j:
                aux += 1
        acc = aux / n
        return acc

    def getNN1st(self, x_qi, kd):
        self.NN1st = []
        self.NN1st = self._calculatekNN(x_qi, kd) # INDICES
        #self.NN1st = np.array([self.X_train[i] for i in k_i])
        
        #self.distances = [self.getMetric(x_q, i) for i in self.NN1st]
        self.distances = [self.Dmatrix[x_qi, i] for i in self.NN1st]

    def getNN2nd(self, x_qi, kd):
        self.NN2nd = []

        k_i = [self._calculatekNN(i, kd+1) for i in self.NN1st]
        #print('kis: ',k_i)
        #NN2nd_aux = [self.X_train[j[1:]] for j in k_i]
        NN2nd_aux = [j[1:] for j in k_i]
        #print('2nn aux:', NN2nd_aux)

        #firstRatio = self.getMetric(x_q, self.NN1st[-1])
        firstRatio = self.Dmatrix[x_qi, self.NN1st[-1]]

        #for idx in NN2nd_aux:
        for idx, k in enumerate(NN2nd_aux):
            #dist = [self.getMetric(x_q, j) for j in k] # para no considerarse a si mismo
            dist = [self.Dmatrix[x_qi, j] for j in k]
            k_eff_i = np.where(dist < 2*firstRatio)
            k = k[k_eff_i[0]]
            k = np.insert(k, 0, self.NN1st[idx], axis = 0) # insertamos el nn1st para el calculo de centroides (indice 0)

            self.NN2nd.append(k)
        #print('2nn:', self.NN2nd)
    
    def centroids(self):
        _centroids = []
        for neighbors in self.NN2nd:
            aux_sums = []
            N = len(neighbors)
            for i in range(0, N):
                aux = 0
                for j in range(0,N):
                    aux += self.Dmatrix[neighbors[i], neighbors[j]]
                
                aux_sums.append(aux)
            c_i = np.argsort(aux_sums)[0]
            centroid = neighbors[c_i]
            _centroids.append(centroid)
            
           
        #aux_sums = [np.sum(self.Dmatrix[i,0:self.m1]) for i in self.NN2nd]
        #centroids_ = [np.sum(i, axis = 0)/len(i) for i in self.NN2nd]
        return _centroids

    def getNNext(self, x_qi):
        centroids = self.centroids()
        dist = [self.Dmatrix[x_qi, j] for j in centroids]
        self.NNext = []
        
        for i in range(len(dist)):
            if dist[i]<=self.distances[i]:
                for j in self.NN2nd[i]:
                    self.NNext.append(j)
            else:
                self.NNext.append(self.NN1st[i])

        self.NNext = np.array(self.NNext)

    def backwards_knn(self, x_i, x_qi, kb, data = []): # x_i es un indice
        distances = self.Dmatrix[x_i, 0:self.m1]
        distances = np.concatenate((distances, [self.Dmatrix[x_i, x_qi]]))

        k_i = np.argsort(distances)[:int(kb+1)]
        if len(distances)-1 in k_i:
            return True
        else:
            return False


    def getNNtwo(self, x_qi, kd):
        kb = np.ceil(self.kb_factor*kd)
        self.NNext = np.unique(self.NNext, axis = 0)
        #print('NNext unique: ', self.NNext)
        self.NNtwo = []
        #X_train_star = np.concatenate((self.Dmatrix[0,self.m1,:], self.Dmatrix[x_qi,:]), axis = 0)
        #X_train_star = np.concatenate((self.X_train, x_q.reshape(1,self.n)), axis = 0)
        #k_i_two = [self._calculatekNN(i, kb+1, data = X_train_star) for i in self.NNext] #+1 pporque se considera a si mismo
        
        #NNtwo_aux = [X_train_star[two] for two in k_i_two]
        for ni in self.NNext:
            if self.backwards_knn(ni, x_qi, kb):
                self.NNtwo.append(ni)
        self.NNtwo = np.array(self.NNtwo)

    def votes(self):
        if len(self.NNtwo) == 0:
            k_labels = [self.y_train[label] for label in self.NN1st]
        else:
            k_labels = [self.y_train[label] for label in self.NNtwo]
        values, counts = np.unique(k_labels, return_counts = True)
        ind = np.argmax(counts)
        tag = values[ind]
        return tag

    
if __name__ == '__main__':
    dataset = pd.read_csv('Dataset_vocesp.csv')
    
    X = dataset.drop("Clase", axis=1).values
    y = dataset["Clase"].values

    # dividimos datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #12

    classifier = TLKDNN()
    classifier.fit(X_train, y_train, X_test, y_test)


    preds = classifier.predict(X_test)
    print('preds: ', preds)
    print('test: ', y_test)
    print('acc: ', classifier.accuracy(np.array(preds), y_test))
