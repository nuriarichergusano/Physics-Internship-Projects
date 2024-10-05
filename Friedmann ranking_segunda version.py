#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:38:08 2024

@author: nuriarichergusano
"""

from scipy.stats import rankdata
import pandas as pd
from numpy import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def print_results(prefix, y_true, y_pred, C=None): 
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f'{prefix}: kappa={kappa:.1%} ')
    cf = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:')
    print(cf)
    return kappa  # Devuelve kappa

def friedman_rank(perf):
    [nmodel, ndata] = perf.shape
    pos = zeros([nmodel, ndata])
    for i in range(ndata):
        # Orden descendente
        ind = argsort(perf[i, :])[::-1]
        for j in range(nmodel):
            pos[i, ind[j]] = j + 1
    print(pos)
    fr = mean(pos, axis=0)
    return fr

classifiers = ['SVM', 'MLP', 'LDA', 'RF']

archivos_datasets = ['wine.txt', 'hepatitis.txt', 'annealing.txt', 'heart-disease.txt']

kappa_values_methodology_1 = []

kappa_values_methodology_2 = []

for dataset in archivos_datasets:
    nf = dataset
    x = loadtxt(nf) 
    y = x[:,0] - 1
    x = delete(x, 0, 1)
    C = len(unique(y))
    print('Dataset %s' % dataset)
    
    dataset_kappa_methodology_1 = []
    dataset_kappa_methodology_2 = []

    # -----------------------------------------------
    # METODOLOGÍA 1 : TRAIN+TEST CONJUNTO COMPLETO
    # -----------------------------------------------
    for classifier in classifiers:
        if classifier == 'LDA':
            lda = LinearDiscriminantAnalysis().fit(x, y)
            z_lda = lda.predict(x)
            kappa = print_results('Train + test conjunto total (LDA)', y, z_lda)
        elif classifier == 'MLP':
            neurons = [20, 20, len(unique(y))]
            mlp = MLPClassifier(hidden_layer_sizes=neurons).fit(x, y)
            z_mlp = mlp.predict(x)
            kappa = print_results('Train+Test  conjunto total (MLP)', y, z_mlp, C=len(unique(y)))
        elif classifier == 'RF':
            rf = RandomForestClassifier(n_estimators=20).fit(x, y)
            z_rf = rf.predict(x)
            kappa = print_results('Train+Test (Random Forest)', y, z_rf, C=len(unique(y)))
        elif classifier == 'SVM':
            svm = SVC().fit(x, y)
            z_svm = svm.predict(x)
            kappa = print_results('Train+Test (SVM)', y, z_svm, C=len(unique(y)))
        dataset_kappa_methodology_1.append(kappa)  # Agregar kappa a la lista de este dataset

    # -----------------------------------------------
    # METODOLOGÍA 2 : TEST 30% + TRAIN 70%
    # -----------------------------------------------
    for classifier in classifiers:
        if classifier == 'LDA':
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=180)
            lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
            z_lda = lda.predict(x_test)
            kappa = print_results('Train+Test 70/30 (LDA)', y_test, z_lda, C=len(unique(y)))
        elif classifier == 'MLP':
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)
            mlp = MLPClassifier(hidden_layer_sizes=20).fit(x_train, y_train) 
            zm_mlp = mlp.predict(x_test)
            kappa = print_results('Train+Test 70/30 (MLP)', y_test, zm_mlp, C=len(unique(y)))
        elif classifier == 'RF':
            x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(x, y, test_size=0.3, random_state=180)
            rf_model = RandomForestClassifier(n_estimators=20).fit(x_train_rf, y_train_rf)
            z_rf = rf_model.predict(x_test_rf)
            kappa = print_results('Train+Test 70/30 (Random Forest)', y_test_rf, z_rf, C=len(unique(y)))
        elif classifier == 'SVM':
            x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x, y, test_size=0.3, random_state=300)
            svm_model = SVC().fit(x_train_svm, y_train_svm)
            z_svm = svm_model.predict(x_test_svm)
            kappa = print_results('Train+Test 70/30 (SVM)', y_test_svm, z_svm, C=len(unique(y)))
        dataset_kappa_methodology_2.append(kappa)  # Agregar kappa a la lista de este dataset

    kappa_values_methodology_1.append(dataset_kappa_methodology_1)  # Agregar la lista de kappa de este dataset a la lista principal
    kappa_values_methodology_2.append(dataset_kappa_methodology_2)  # Agregar la lista de kappa de este dataset a la lista principal

for i, kappa_values in enumerate([kappa_values_methodology_1, kappa_values_methodology_2]):
    with open(f'kappa_values_methodology_{i + 1}.txt', 'w') as f:
        for row in kappa_values:
            f.write('\t'.join(map(str, row)) + '\n')

for i, kappa_values in enumerate([kappa_values_methodology_1, kappa_values_methodology_2]):
    # Leer los valores de kappa del archivo de texto
    with open(f'kappa_values_methodology_{i + 1}.txt', 'r') as f:
        lines = f.readlines()
    
    kappa_matrix = []
    for line in lines:
        kappa_row = list(map(float, line.strip().split('\t')))
        kappa_matrix.append(kappa_row)
    kappa_matrix = array(kappa_matrix)
    
    friedman_ranks = friedman_rank(kappa_matrix)
    
    print(f"Metodología {i + 1} - Friedman Ranks:")
    for j, rank in enumerate(friedman_ranks):
        print(f"Classifier {classifiers[j]}: {rank}")
        

with open('kappa_values_methodology_1.txt', 'r') as f:
    lines_methodology_1 = f.readlines()

kappa_matrix_methodology_1 = []
for line in lines_methodology_1:
    kappa_row = list(map(float, line.strip().split('\t')))
    kappa_matrix_methodology_1.append(kappa_row)
kappa_matrix_methodology_1 = array(kappa_matrix_methodology_1)

with open('kappa_values_methodology_2.txt', 'r') as f:
    lines_methodology_2 = f.readlines()

kappa_matrix_methodology_2 = []
for line in lines_methodology_2:
    kappa_row = list(map(float, line.strip().split('\t')))
    kappa_matrix_methodology_2.append(kappa_row)
kappa_matrix_methodology_2 = array(kappa_matrix_methodology_2)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axs[0].boxplot(kappa_matrix_methodology_1.T)
axs[0].set_title('Metodología 1')
axs[0].set_xticklabels(classifiers)
axs[0].set_ylabel('Kappa')

axs[1].boxplot(kappa_matrix_methodology_2.T)
axs[1].set_title('Metodología 2')
axs[1].set_xticklabels(classifiers)

plt.tight_layout()
plt.show()

'Están intercambiados'
