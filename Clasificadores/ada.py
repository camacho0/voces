import numpy as np
class DKNN:
    def fit(self, X_train, y_train, n_samples):
        self.X = X_train
        self.y = y_train
        np.random.seed(9)
        self.n_samples = n_samples
        self.m_samples, _ = X_train.shape
        self.k1_max = int(np.ceil(self.m_samples ** (1/3)))
        self.k_med, self.trainK_, self.pClass, self.min, self.max = self.ada_knn2(self.X, self.y)

    def ada_knn2(self, X_train, y_train):
        k_max = int(np.ceil(np.sqrt(self.m_samples)))

        k_new = np.random.permutation(k_max)+1
        k_med_new = np.zeros((self.n_samples,1))

        trainK = []
        nnDist = np.zeros((self.m_samples, 1))
        self.classes, counts = np.unique(y_train, return_counts = True)
        if self.classes[0] == 0:
            self.aug_classes = np.append(self.classes, len(self.classes))
        else: 
            self.aug_classes = np.append(self.classes, len(self.classes) + self.classes[0]) # CHECAR
        n_classes = len(self.classes)
        counts = counts.reshape(len(counts), 1)
        k_alpha = min(10, k_max)
        
        pClass = np.ones((n_classes,1))
        ## Weighted neighbors (GIHS) ##
        #pClass = self.GIHS(n_classes, self.classes, counts, k_max)

        for i in range(self.m_samples):
            flag = 0
            km = []

            distanceM = np.array([[np.linalg.norm(X_train[0,:] - X_train[i,:])]
                                  for i in range(1,self.m_samples)])
            distanceI = np.argsort(distanceM, axis = 0)
            #distanceM = np.sort(distanceM, axis = 0)
            nnDist[i] = distanceM[distanceI[0]]
            h = 0
            for i in range(k_alpha):
                h = self.kNNImb(y_train[1:], k_new[i], distanceI, self.classes, pClass)
                if h == y_train[0]:
                    flag = 1
                    #km = np.append(km, k_new[i])
                    km.append(k_new[i])

            if flag != 1:
                for j in range(10, k_max):
                    h = self.kNNImb(y_train[1:], k_new[j], distanceI, self.classes, pClass)
                    if h == y_train[0]:
                        km.append(k_new[i])
                        break

            if len(km) == 0:
                km = [np.random.randint(10) + 1]

            trainK.append(km)
            #trainK += km
            X_train = np.roll(X_train, 1, axis = 0)
            y_train = np.roll(y_train, 1, axis = 0)

        minDist = np.min(nnDist)
        maxDist = np.max(nnDist)

        return k_med_new, trainK, pClass, minDist, maxDist

    def predict(self, X_test, y_test):
        p_labels = np.zeros((self.n_samples, 1))
        for i in range(self.n_samples):
            x0 = X_test[i,:]
            distanceM = np.array([np.linalg.norm(x0 - self.X[i,:])
                                  for i in range(self.n_samples)])
            distanceI = np.argsort(distanceM)#, axis = 0)
            #distanceM = np.sort(distanceM)#, axis = 0)
            nn_test_dist = distanceM[distanceI[0]]
            self.k_med[i] = self.learning_model(distanceI, self.trainK_, self.m_samples, nn_test_dist,
                                              self.min, self.max, self.k1_max)
            p_labels[i] = self.kNNImb(self.y, self.k_med[i], distanceI, self.classes, self.pClass)

        #return p_labels

    def learning_model(self, Idx, k_value, a, d_minY, d_minTr, d_maxTr, K1max):
        k = 0
        if d_minY <= d_minTr:
            k = K1max
        elif d_minY >= d_maxTr:
            k = 1
        elif d_minY > d_minTr and d_minY < d_maxTr:
            beta = -np.log(K1max) / (d_maxTr - d_minTr)
            k_lin = ((1-K1max)/(d_maxTr-d_minTr))*(d_minY-d_minTr) + K1max
            k_exp = K1max * np.exp((d_minY - d_minY) * beta)

            k = int(np.ceil(np.sqrt(k_lin * k_exp)))

        nu = int(np.ceil(np.sqrt(a)))
        i_nearest = Idx[:k]
        #if len(i_nearest)<k:
        #  k = len(i_nearest)
        #con_arr = np.array([k_value[i_nearest[j]] for j in range(k)])
        con_arr = []
        for i in range(k):
            con_arr += k_value[i_nearest[i]]
        #con_arr = [[] + k_value[i_nearest[j]] for j in range(k)]
        max_count = np.histogram(con_arr, bins = np.arange(nu+1))[0]
        max_fr = np.max(max_count)
        all_possible_k = [j for j in range(nu) if max_count[j] == max_fr]
        all_possible_k.append(np.argmax(max_count))
        all_possible_k = np.unique(all_possible_k)
        pos = np.random.randint(len(all_possible_k))
        k_val_T = all_possible_k[pos]
        
        return k_val_T

    def GIHS(self, classNum, classes, actual, k_max):
        ## GIHS ##
        ideal = np.ones((classNum, 1)) / classNum
        actual = actual / self.m_samples
        zeroActual = actual == 0
        pClass_ = np.ones((classNum, 1))
        change = np.zeros((classNum, 1))
        change[~zeroActual] = (ideal[~zeroActual] - actual[~zeroActual]) / actual[~zeroActual]
        pClass_ += change

        return pClass_, k_alpha

    def kNNImb(self, tlabel, k1, distanceI, all_label, pClass):
        k = int(k1)
        k_neighbor = distanceI[:k]#.reshape(1,k)
        neighbor_label = tlabel[k_neighbor]#.reshape((k,1))
        #_, h = np.unique(neighbor_label, return_counts = True)
        h, _ = np.histogram(neighbor_label, bins = self.aug_classes)
        h = h.reshape((len(h),1))*pClass
        maxH = np.max(h)
        maxI = h == maxH
        maxI = maxI * pClass
        label_pos = np.argmax(maxI)
        p_label = all_label[label_pos]

        return p_label
