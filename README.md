# Deteccion de Deterioro Cognitivo a traves de voz
En este proyecto se clasifica entre pacientes con deterioro cognitivo y sanos, mediante voz. Para ello se extrajeron caracteristicas de audios de pacientes proporcionados por investigadores del IMSS y sobre ellas se evaluo el desempeno de KNN, Random Forest , MLP y SVM. 
## Dataset
El dataset  no se puede proporcionar por cuestiones de privacidad. Sin embargo, se proporcionan 3 csv con la extraccion de caracteristicas hecha de los audios ya filtrados de ruido y voces de medicos y divididos en segmentos de 3s. 

## Resultados
Se obtuvieron los siguientes resultados con los clasificadores utilizando leave one out
|  Clasificador | Accuracy   | F1   | Recall   |
| :------------: | :------------: | :------------: | :------------: |
|  KNN |  0.54 &plusmn;  0.31| 0.43&plusmn; 0.44  | 0.39 &plusmn; 0.39  |
|  MLP | 0.79  | 0.84   | 0.91  |
|   SVM|  0.85&plusmn; 0.28 |0.46  &plusmn;0.47  |0.44&plusmn; 0.47   |
|  Random Forest |0.81 &plusmn; 0.18    | 0.46 &plusmn; 0.46  | 0.43 &plusmn; 0.43  |
