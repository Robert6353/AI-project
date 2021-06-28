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
from matplotlib import pyplot
from numpy import argmax, sqrt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#Convert the data and target into test, train and validation set
x_train = pd.read_csv("x_smote.csv", index_col=0)

#ic(x_train)

#Get the target value for the smtoe dataset
y_train = pd.read_csv("y_smote.csv", index_col = 0)

#ic(y_train)

x_train.info()

x_test = pd.read_csv("x_test_nsmote.csv", index_col = 0)

#ic(x_test)

y_test = pd.read_csv("y_test_nsmote.csv", index_col = 0)

#print(y_test)

#ic(y_test)

print(x_train.columns)

def feature(clf):
    feature_importance = abs(clf.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    print(pos)
    
    featfig = pyplot.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(x_train.columns)[sorted_idx], fontsize=8)
    featax.set_xlabel('Relative Feature Importance')
    
    pyplot.tight_layout()   
    pyplot.show()

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
    threshold = 0.1

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(threshold)
    print(accuracy)
    #print(report)
    sns.heatmap((accuracy), annot=True, fmt='')
    

def log_regress():
    name = "Logistic regression"
    
    clf = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x_train,y_train)
    
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    #feat_importances.nlargest(10).plot(kind='barh')
    #plt.show()
    
    #print(accuracy)
    #print(report)
    
    #sns.heatmap(accuracy, annot=True)

    #feature(clf)
    
    #percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
    return feat_importances
    
log_regress()

def knn_regression():
    name = "KNN_regression"
    
    clf = KNeighborsClassifier(n_neighbors=15)
    
    clf.fit(x_train, y_train)

    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.1

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x_train,y_train)
    
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    #feat_importances.nlargest(10).plot(kind='barh')
    #plt.show()
    
    #feature(clf)
    
    #percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
    return feat_importances
    
knn_regression()

def random_forest():
    name = "random forest"
    
    clf = RandomForestClassifier(max_depth=100, random_state=0)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5
    
    #labels = list(x_train.columns.values)

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x_train,y_train)
    
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    #feat_importances.plot(kind='barh')
    #plt.show()
    
    #percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
    return feat_importances

random_forest()

def SVM():
    name = "Support vector machines"
    
    clf = svm.SVC(probability = True)
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.5

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x_train,y_train)
    
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    #feat_importances.nsmallest(10).plot(kind='barh')
    #plt.show()
    
    #feature(clf)
    
    #percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
    return feat_importances

SVM()

def dec_tree():
    name = "decision tree"
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    y_true = y_test.replace({"b'0'": 0, "b'1'": 1})
    
    threshold = 0.3

    predicted_proba = clf.predict_proba(x_test)
    
    predicted = (predicted_proba[:,1] >= threshold).astype('int')

    accuracy = confusion_matrix(y_true, predicted)
    report = classification_report(y_true, predicted)
    
    print(accuracy)
    print(report)
    
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(x_train,y_train)
    
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    #feat_importances.plot(kind='barh')
    #plt.show()
    
    #feature(clf)
    
    #percent(clf, y_true)
    
    roc_curves(clf, name, accuracy)
    
    return feat_importances
dec_tree()

#feat_importance_dec_tree = dec_tree()

#feat_importance_SVM = SVM()

#feat_importance_rand = random_forest()

#feat_importance_knn = knn_regression()

#feat_importance_log = log_regress()

#total = feat_importance_dec_tree.add(feat_importance_SVM)

#total = total.add(feat_importance_rand)

#total = total.add(feat_importance_knn)

#total = total.add(log_regress())

#print(feat_importance_SVM)

#print(total)

#total.plot(kind='barh')
#plt.show()






