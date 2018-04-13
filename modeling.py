#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:48:13 2018

@author: miller
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import *
from sklearn.cross_validation import KFold

### Train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_train)

def get_acc_auc_kfold(X,Y,k=5):
        
	#Report the mean accuracy and mean auc of all the folds for logistic regression model
            
    kf = KFold(X.shape[0],k)
    
    acc_dict = {}
    auc_dict = {}
    
    for regularization_wt in [1,10,100,0.1]:
    
        accuracy_list = []
        auc_list = []
    
        for train_index, test_index in kf:
                    
            # Separating train/test sets
            x_train = X[np.array(train_index)]
            x_test = X[np.array(test_index)]
            
            y_train = Y[np.array(train_index)]
            y_test = Y[np.array(test_index)]
            
            # Training model
            lr = LogisticRegression(C = regularization_wt, penalty="l2")
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)  
                        
            accuracy_list.append(accuracy_score(y_test, y_pred) )
            auc_list.append(roc_auc_score(y_test, y_pred) )
            
        acc_dict[regularization_wt] = np.mean(accuracy_list)
        auc_dict[regularization_wt] =  np.mean(auc_list)
            
    return auc_dict, acc_dict