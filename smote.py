import pandas as pd
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
#from data_prep import drop_col

def read_csvs():
    pol_original = pd.read_csv("pol.csv", index_col=0)

    print(pol_original)

    pol = pd.read_csv("pol_new.csv")
    
    pol_y = pol_original["class"]
    
    pol_y.to_csv("pol_y.csv")
    
    pol_y = pd.read_csv("pol_y.csv")
    
    pol = pol.merge(pol_y)
    
    print(pol)
    
    pol = pol.drop(pol.columns[0], axis=1)
    
    print(pol)

    return pol, pol_original

def train_test(df, df_original):
    train, test = train_test_split(df, test_size=0.2)

    print("Train Bitch")
    print(train)

    print("Test Bitch")
    print(test)

    #Resplit the train dataset into the inputs and targt values again
    x_train = train.iloc[:,0:26]

    y_train = train["class"]

    print("x_train")
    print(x_train)

    print("y_train")
    print(y_train)

    #Resplit the train datset into the inpts and target values again
    x_test = test.iloc[:,0:26]

    y_test = test["class"]

    print("x_test")
    print(x_test)

    print("y_test")
    print(y_test)
    
    return train, test, x_train, y_train, x_test, y_test

def standardise(pol):
    #Standardisation
    from sklearn import preprocessing

    #std_scale = preprocessing.normalize(pol_transt.iloc[:,0:64], norm='l1', axis=0, copy=True)

    cols = [pol.columns]
    
    index = [pol.index]

    std_scale = preprocessing.StandardScaler().fit(pol)

    pol_std = std_scale.transform(pol)
    
    pol = pd.DataFrame(pol_std, columns = cols, index = index)
    
    print(pol)
    
    return pol

def check_data_imbalance(pol):
    print(pol.groupby('class').size())
    minority_percent = (pol['class'].tolist().count(1) / len(pol['class'].tolist()))*100
    print('Minority (label 1) percentage: '+  str(minority_percent) + '%')
        
#check_data_imbalance(train)

# =============================================================================
# def split_dataframes_features_labels(dfs):
#     feature_dfs = [dfs.iloc[:,28]]
#     label_dfs = [dfs.iloc[:,28]]
#     print("")
#     print( feature_dfs)
#     print("")
#     print(label_dfs)
# =============================================================================

#split_dataframes_features_labels(pol)

def smote(x_train, y_train):
    X = pd.DataFrame(x_train)
    #print(X)
    y = y_train
    undersample = RandomUnderSampler(sampling_strategy = 0.1)
    
    print("X")
    print(X)
    
    print("y")
    print(y)
    
    print(Counter(y))

    #smote = SMOTE(random_state=0)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    
    #print(X_resampled)

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    #print(X_resampled)

    print(Counter(y_resampled))
    print(y_resampled)

    oversample = SMOTE(sampling_strategy = 0.2)

    X_Smote, y_Smote = oversample.fit_resample(X_resampled, y_resampled)
    
    return X_Smote, y_Smote

def smote_csv(X_Smote, y_Smote):
    #print(X_Smote)
    #print("")
    #print(y_Smote)

    print(Counter(y_Smote))

    #X_Smote.to_csv("x_smote.csv")

    #y_Smote.to_csv("y_smote.csv")

df, df_original = read_csvs()

train, test, x_train, y_train, x_test, y_test = train_test(df, df_original)

x_train = standardise(x_train)

print("x_train")
print(x_train)

print("y_train")
print(y_train)

#x_train.to_csv("x_train_nsmote.csv")

#y_train.to_csv("y_train_nsmote.csv")

x_test = standardise(x_test)

print("x_test")
print(x_test)

#x_test.to_csv("x_test_nsmote.csv")

#y_test.to_csv("y_test_nsmote.scv")

#df1 = x_train.assign(e=pd.Series(np.random.randn(sLength)).values)

#pol = x_train.assign(bank = y_train[0])

#print("pol")
#print(pol)

x_smote, y_smote = smote(x_train, y_train)

print(x_smote)

print(y_smote)

smote_csv(x_smote, y_smote)

#pol.to_csv("pol_smote_train.csv")

#x_s_train = smote(x_train)

#print(x_s_train)

#x_s_test = smote(x_test)






