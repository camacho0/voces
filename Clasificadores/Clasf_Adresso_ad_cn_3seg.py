# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:05:48 2023

@author: MigGius
"""

import os
from os import remove
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import opensmile
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

#%%

dir_pp = "C:/Users/MigGius/Documents/ADReSSo/ADReSSo/diagnosis/Segm/"
path =[]
# Cargar el archivo de audio
#path.append( "test-dist/audio/")
path.append( "train/audio/cn/audios1/") #Clase 0  --- CN Control
path.append( "train/audio/cn_enhanced/audios3/") #Clase 2-0  --- CN Control
path.append( "train/audio/ad/audios0/") #Clase 1  --- AD Demencia Alzheimer
path.append( "train/audio/ad_enhanced/audios2/") #Clase 3-1  --- AD Demencia Alzheimer
open_smile = pd.DataFrame()
tipo = pd.DataFrame()
dc = []
for idx, fold in enumerate(path):
    archivos = os.listdir(f'{dir_pp}{fold}')
    if 'desktop.ini' in archivos:
        archivos.remove('desktop.ini')
    for f in archivos:
        smile = opensmile.Smile( feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals, )
        op_sm = smile.process_file(f'{dir_pp}{fold}{f}')
        open_smile = pd.concat([open_smile, op_sm], axis=0, ignore_index=True)
        dc.append(idx//2)
        aux=pd.DataFrame(archivos)
        aux.to_csv(f'archivos_3seg_{idx}.csv')
tipo['Deterioro'] = dc
open_smile = pd.concat([open_smile, tipo], axis=1)

open_smile.to_csv('Op_Sm_88c_3seg_Sil.csv')

#%% PAR (SIN RUIDO) E IMPAR (CON RUIDO)

#op_sm_con = open_smile.iloc[0:1773, :] 
#op_sm_con = op_sm_con.append(open_smile.iloc[4271:6044, :])
#op_sm_sin = open_smile.iloc[1773:4271, :] 
#op_sm_sin = op_sm_sin.append(open_smile.iloc[6044:, :])

op_sm_con = open_smile.iloc[0:1201, :] 
op_sm_con = op_sm_con.append(open_smile.iloc[2428:3869, :])
op_sm_sin = open_smile.iloc[1201:2428, :] 
op_sm_sin = op_sm_sin.append(open_smile.iloc[3869:, :])

lst_df = [open_smile, op_sm_sin, op_sm_con]
#%% ######### -------- CLASIFICACION SVM
mod_svm = SVC(kernel='rbf')
mod_knn = KNeighborsClassifier(n_neighbors=5)
mod_rdf = RandomForestClassifier()

lst_eva = ['f1', 'accuracy', 'recall', 'precision']

prom_res=[]
for df in lst_df: #Se va a iterar con los tres tipos de datos, mezclados, con y sin ruidos
    scores_svm = cross_validate(mod_svm, df.iloc[:,:-1], df.iloc[:,-1], scoring=lst_eva, cv=10)
    scores_knn = cross_validate(mod_knn, df.iloc[:,:-1], df.iloc[:,-1], scoring=lst_eva, cv=10)
    scores_rdf = cross_validate(mod_rdf, df.iloc[:,:-1], df.iloc[:,-1], scoring=lst_eva, cv=10)
    
    del scores_svm['fit_time'], scores_svm['score_time'], scores_knn['fit_time'], scores_knn['score_time'], scores_rdf['fit_time'], scores_rdf['score_time']
    
    
    prom_svm = []
    for clave, valor in scores_svm.items():
        prom_svm.append([clave, np.mean(scores_svm[clave])])
        
    prom_knn = []
    for clave, valor in scores_knn.items():
        prom_knn.append([clave, np.mean(scores_knn[clave])])
        
    prom_rdf = []
    for clave, valor in scores_rdf.items():
        prom_rdf.append([clave, np.mean(scores_rdf[clave])])
    
    prom_res.append([prom_svm, prom_knn, prom_rdf])
    
prom_mix =prom_res[0]
prom_sin =prom_res[1]
prom_con =prom_res[2]
