import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix, 
                           accuracy_score, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from icecream import ic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from numpy import argmax, sqrt


#Convert the data and target into test, train and validation set
#--------------------------------------------------------------------------
#Get the input values for smote dataset
x_train = pd.read_csv("x_train_nsmote.csv", index_col=0)

#ic(x_train)

#Get the target value for the smtoe dataset
y_train = pd.read_csv("y_train_nsmote.csv", index_col = 0)

#ic(y_train)

x_test = pd.read_csv("x_test_nsmote.csv", index_col = 0)

#ic(x_test)

y_test = pd.read_csv("y_test_nsmote.csv", index_col = 0)

#ic(y_test)

def roc_curves(clf, name, accuracy):
    y_probs = clf.predict_proba(x_test)
    
    ns_probs = [0 for _ in range(len(y_test))]
    
    # keep probabilities for the positive outcome only
    y_probs = y_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, y_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print(str(name) + ': ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, threshold = roc_curve(y_test, ns_probs, pos_label = "b'1'")
    lr_fpr, lr_tpr, threshold = roc_curve(y_test, y_probs, pos_label = "b'1'")
    
    aca = 0.5*(lr_tpr * (1-lr_fpr))
    
    ix = argmax(aca)
    
    print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], aca[ix]))
    
    # plot the roc curve for the model
    #pyplot.plot(ns_fpr, ns_tpr, label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, label= name)
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

def percent(clf, y_true):
    threshold = 1

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(threshold)
    print(accuracy)
    print(report)

def log_regress():
    name = "logistic regression"
    
    clf = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
#log_regress()

def knn_regression():
    name = "knn regression"
    
    clf = KNeighborsClassifier(n_neighbors=5)
    
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
#knn_regression()

def random_forest():
    name  = "random forest"
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)

random_forest()

def SVM():
    name = "Support vector machines"
    
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)

#SVM()

def dec_tree():
    name = "decision tress"
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)

dec_tree()