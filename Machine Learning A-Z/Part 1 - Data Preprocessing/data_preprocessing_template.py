# -*- coding: utf-8 -*-

# Importing  libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values ##variables independientes, todas las columnas menos la ultima
y = dataset.iloc[:, 3].values #variables dependientes

#Splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,
                                                     random_state = 0)
  #feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
 #para el training set hay que fitearlo antes de transformarlos
x_test = sc_x.transform(x_test)"""
  
