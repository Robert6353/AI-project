import numpy as np
import os
import pandas as pd
from scipy.io.arff import loadarff 
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from impyute.imputation.cs import mice
#from data_viz import missing_columns, missing_corr

def convert_data():
    raw_data = loadarff('5year.arff')
    pol = pd.DataFrame(raw_data[0])

def xandy(pol):
    pol_y = pol["class"]

    pol_y.to_csv("pol_y.csv")

    pol_y = pd.read_csv("pol_y.csv", index_col=0)

def CSV():
    #Convert into a csv file to be able to visualize better(personal preference)
    if os.path.isfile("pol.csv"):
        pass
    else:
        pol.to_csv("pol.csv")

def mice():
    imputer = IterativeImputer(max_iter=int(1000))

    imputer.fit(pol.iloc[:,0:64])

    pol_trans = imputer.transform(pol.iloc[:,0:64])

    print('Missing: %d' % sum(np.isnan(pol_trans).flatten()))

    print(pol_trans)

    cols = ['Attr' + str(i+1) for i in range(len(pol.columns)-1)]

    pol_transt = pd.DataFrame(pol_trans, columns = cols)

    print(pol_transt)

    print(pol.iloc[:,63])
    
    pol_transt.to_csv("pol_impute.csv")
    
def standardise(pol):
    #Standardisation
    from sklearn import preprocessing

    #std_scale = preprocessing.normalize(pol_transt.iloc[:,0:64], norm='l1', axis=0, copy=True)

    cols = ['Attr' + str(i+1) for i in range(len(pol.columns))]

    std_scale = preprocessing.StandardScaler().fit(pol)

    pol = std_scale.transform(pol)
    
    pol = pd.DataFrame(pol, columns = cols)
    
    print(pol)
    
    pol.to_csv("pol_standardise_x.csv")

def heatmap_matrix(pol_impute):
    #pol = pol.drop(columns = ["Attr28", "Attr41", "Attr43", "Attr32", "Attr59", "Attr52",
    #        "Attr5", "Attr47", "Attr15", "Attr4", "Attr30", "Attr20", "Attr57"])
    
    cor_matrix = pol_impute.corr().abs()

    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1)
                         .astype(np.bool))
    
    drop_col(upper_tri, cor_matrix, pol_impute)

def drop_col(upper_tri, cor_matrix, pol_impute):
    to_drop = [column for column in upper_tri.columns
               if any(upper_tri[column] > 0.70)]
    #print(to_drop)

    #Has a colinearity of 0.5
    #print(cor_matrix.iloc[0:1, 6:7])
    
    upper_tri = upper_tri.drop(columns = to_drop)
    
    sns.heatmap(upper_tri.iloc[1:66, 1:66])

    #pol_new = pol_impute.drop(to_drop, axis = 1)
    
    #pol_new_csv(pol_new, pol)
    
def pol_new_csv(pol_new, pol):
    print(pol_new.head())
    print(pol)

    missing_columns(pol_new)

    missing_corr(pol_new)

    pol_new.to_csv("pol_new.csv")

#pol = pd.read_csv("pol.csv")

#mice()

pol_impute = pd.read_csv("pol.csv")

cor_matrix = pol_impute.corr().abs()

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1)
                     .astype(np.bool))

to_drop = [column for column in upper_tri.columns
           if any(upper_tri[column] > 0.70)]

upper_tri = pol_impute.drop(columns = to_drop)

sns.heatmap(upper_tri.iloc[1:20, 1:20])

#heatmap_matrix(pol_impute)

#standardise(pol_impute)

