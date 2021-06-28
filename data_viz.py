import os
import pandas as pd
from scipy.io.arff import loadarff
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = loadarff('5year.arff')
pol = pd.DataFrame(raw_data[0])

print(pol.head())
print(pol.tail())

df = pol.iloc[:,0:64]

print(df.iloc[:,56].max() - df.iloc[:,56].min())
print(df.iloc[:,2].max() - df.iloc[:,2].min())

print(df.iloc[:,54].max())

print(df.iloc[:,2].max())

if os.path.isfile("pol.csv"):
    pass
else:
    pol.to_csv("pol.csv")

def bad_reports():
    #Allows me to identify which companies have bad reporting
    pol_tp = pol.transpose()
    print(pol_tp)
    pol_tp_miss = pol_tp.isnull().sum()
    for items in pol_tp_miss.iteritems():
        if items[1] > 5:
            print(items)

def missing_columns(pol):
    #Are there any colums with missing values greater than 100
    pol_missing = pol.isnull().sum()
    for items in pol_missing.iteritems():
        if items[1] > 100:
            print(items)
            #Attribute 37 has over 50% of its values missing, therefore we will get rid of it

def missing_corr(dfs):
    missing_df = dfs.columns[dfs.isnull().any()].tolist()
    msno.heatmap(dfs[missing_df], figsize=(20,20))

def generate_sparsity_matrix(dfs):
    missing_df_i = dfs.columns[dfs.isnull().any()].tolist()
    msno.matrix(dfs[missing_df_i], figsize=(20,5))

#generate_sparsity_matrix(pol)
        
#generate_heatmap(pol)

def corr_heatmap():
    plt.figure(figsize = (25, 25))
    sns.heatmap(pol.corr().abs())

def correlation():
    pol.corr().abs().to_csv("corr.csv")