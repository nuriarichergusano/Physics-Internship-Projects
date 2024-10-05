#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:51:07 2024

@author: nuriarichergusano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.svm import SVR
from sklearn.metrics import *
from scipy.stats import *



import numpy as np
import pandas as pd

archivo = 'xamon.txt'

data = pd.read_csv(archivo, delimiter='\t')

X = data.iloc[:, 1:-1]  # Todas las columnas excepto la primera y la última
y = data.iloc[:, -1]   # Última columna

print("Características (features):\n", X.head())
print("\nVariable objetivo (output):\n", y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=35)
'Cambiar random_state???? Hacer otro con conjunto completo'


param_grid = {'C': [2**i for i in range(-5, 16)], 'gamma': [2**i for i in range(-10, 11)]}

svr = SVR()


#GRID SEARCH TUNING
grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
'Cambiar cv'

best_params = grid_search.best_params_

best_svr = SVR(C=best_params['C'], gamma=best_params['gamma'])
best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X_test)

correlation = pearsonr(y_test, y_pred)[0] #R
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


y_train_pred_cv = cross_val_predict(best_svr, X_train, y_train, cv=3)
mse_train_cv = mean_squared_error(y_train, y_train_pred_cv)
mse_test_cv = -grid_search.best_score_
variance = mse_test_cv - mse_train_cv
bias = mse_train_cv - mse_test_cv

corr_score = best_svr.score(X_test, y_test) #R^2

print("Correlación R:", correlation)
print("MAE:", mae)
print("RMSE:", rmse)
print("Varianza (V):", variance)
print("Sesgo (Bt):", bias)
print("Coeficiente de correlación (score()):", corr_score)

'Varianza baja: poca desviación respecto de la media'
'Nesgo (sesgo) bajo y negativo???'

# Diagrama de dispersión (scatterplot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Diagrama de Dispersión: Valor Real vs. Predicción')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.show()

# Representación del valor a predecir respecto del número de patrón
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Valor Real', marker='o', linestyle='None')
plt.plot(range(len(y_test)), y_pred, label='Predicción', marker='x', linestyle='None')
plt.title('Valor Real vs. Predicción por Número de Patrón')
plt.xlabel('Número de Patrón')
plt.ylabel('Valor')
plt.legend()
plt.show()


'Ordenar X_train de forma que los outputs estén ordenados de 0 a 6 y pongo uno para entrenamiento y uno para test en lugar de aleatoriamente'
'Filas impares test filas pares train'
'50/50 con validacion cruzada dejando uno fuera'
'Obtener la puntuación final a partir de las parciales'




