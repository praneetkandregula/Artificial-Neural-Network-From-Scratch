import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

#importing original dataset
df = pd.read_csv('LBW_Dataset.csv')

#function to preprocess df
def preprocess(df):
    
    #Drops rows with less than 7 Non-NaN values
    df.dropna(axis = 0,thresh = 7,inplace = True)

    #replace NaN values with mean of the attribute
    mean_age = df['Age'].mean()
    df['Age'].fillna(int(mean_age),inplace = True)
    mean_weight = df['Weight'].mean()
    mean_hb = df['HB'].mean()
    df['HB'].fillna(mean_hb, inplace = True)
    mean_bp = df['BP'].mean()
    df['BP'].fillna(mean_bp, inplace = True)
    df['Weight'].fillna(int(mean_weight), inplace = True)

    #Replace NaN values with mode of the attribute
    mode_del = df['Delivery phase'].mode()[0]
    df['Delivery phase'].fillna(mode_del, inplace = True)
    mode_res = df['Residence'].mode()[0]
    df['Residence'].fillna(mode_res,inplace = True)

    #one hot encoding for categorical data, creates new columns
    comm = pd.get_dummies(df.Community, prefix = "community")
    dphase = pd.get_dummies(df['Delivery phase'],prefix='delphase')
    residence=pd.get_dummies(df.Residence,prefix='res')

    #remove original categorical attributes
    df=df.drop(columns=['Community','Delivery phase','Residence'])
    df=df.join(comm)
    df=df.join(residence)
    df=df.join(dphase)

    #save csv to local disc
    df.to_csv("preprocessed_LBW_Dataset.csv")
    
#function call
preprocess(df)
