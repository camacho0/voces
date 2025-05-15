import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class HVDM:
    def __init__(self, X_train, y_train, cat_ix=[]):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_ix = cat_ix

        # preparamos los rangos de las variables continuas
        self.ranges = np.zeros(shape=(len(self.X_train[0]),))
        for i in range(len(self.ranges)):
            if i not in self.cat_ix:
                self.ranges[i] = 4*np.std(X_train[:,i], axis=0)
        #print(self.ranges)

        # preparemos las constantes de las variables nominales
        self.classes = np.unique(y_train) # obtenemos las clases de y
 
        array_len = 0 # auxiliar para el maximo de valores diferentes de todas las columnas
        for ix in self.cat_ix:
            max_val = len(np.unique(self.X_train[:, ix]))
            if max_val > array_len:
                array_len = max_val

        self.col_ix = [i for i in range(self.X_train.shape[1])] # lista de 0 a n-1 columnas
        self.unique_attributes = np.full((array_len, len(self.col_ix)), fill_value=-1) # matriz de valores distintos por cada columna
        for ix in self.cat_ix:
            unique_vals = np.unique(self.X_train[:, ix])
            self.unique_attributes[0:len(unique_vals), ix] = unique_vals
        #print(self.unique_attributes)

        self.final_count = np.zeros((len(self.col_ix), self.unique_attributes.shape[0], len(self.classes) + 1)) # deja los espacios de los que no son nominales
        # For each column
        for i, col in enumerate(self.cat_ix):
            # For each attribute value in the column
            for j, attr in enumerate(self.unique_attributes[:, col]):
                # If attribute exists
                if attr != -1:
                    # For each output class value
                    for k, val in enumerate(self.classes):
                        # Get an attribute count for each output class
                        row_ixs = np.argwhere(self.X_train[:, col] == attr)
                        cnt = np.sum(self.X_train[row_ixs, self.y_train] == val)
                        self.final_count[col, j, k] = cnt
                    # Get a sum of all occurences
                    self.final_count[col, j, -1] = np.sum(self.final_count[col, j, :])
        #print(self.final_count)



    def hvdm(self, x, y, nan_ix=[]):
        # calculo de la parte continua
        
        aux_cont = []
        for i in range(len(self.ranges)):
            if i not in self.cat_ix: # continuos
                aux_cont.append(abs(x[i]-y[i])/self.ranges[i])
        aux_cont = np.array(aux_cont)

        # calculo de la parte nominal
        result = np.zeros(len(x))
        cat_ix = np.setdiff1d(self.cat_ix, nan_ix)
        
        for i in cat_ix:
            # Get indices to access the final_count array 
            x_ix = np.argwhere(self.unique_attributes[:, i] == x[i]).flatten()
            y_ix = np.argwhere(self.unique_attributes[:, i] == y[i]).flatten()
            # Get the count to calculate the conditional probability
            N_ax = self.final_count[i, x_ix, -1].flatten()
            N_ay = self.final_count[i, y_ix, -1].flatten()
            N_axc = self.final_count[i, x_ix].flatten()
            N_ayc = self.final_count[i, y_ix].flatten()
            if N_ax != 0 and N_ay != 0:
                temp_result = abs(N_axc/N_ax - N_ayc/N_ay)
                temp_result = np.sum(temp_result)
            else:
                temp_result = 0.
            result[i] = temp_result
        
        #print(aux_cont, result)
        #print(np.concatenate([aux_cont, result]))
        return np.sum(np.square(np.concatenate([aux_cont, result])))


                       


if __name__ == '__main__':
    import time

    inicio = time.time()
    time.sleep(2)
    dataset = pd.read_csv('Datasets/tae.csv')
    
    X = dataset.drop('class', axis=1).values
    y = dataset['class'].values

    # dividimos datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #12

    # instanciamos una metrica de tipo HVDM
    metric = HVDM(X_train, y_train, [0,1,2,3])
    print('dis =', metric.hvdm(X_train[0], X_train[1]))
    fin = time.time()

    print(fin-inicio)

