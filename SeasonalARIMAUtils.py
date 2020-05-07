# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column.
"""
def importSeasonalARIMADataset(seasonalARIMADatasetFileName):
    
    seasonalARIMADataset = pd.read_csv(seasonalARIMADatasetFileName)
    
    # Add a "date" datetime column
    seasonalARIMADataset['date']=pd.to_datetime(dict(year=seasonalARIMADataset['year'], month=seasonalARIMADataset['month'], day=1))
    
    # Set "date" to be the index
    seasonalARIMADataset.set_index('date',inplace=True)
    
    #the dataset is monthly dataset. Hence setting its frequency as monthly.
    seasonalARIMADataset.index.freq = "MS"
    
    return seasonalARIMADataset

#splitting dataset into training and testing set
def splitSeasonalARIMADataset(seasonalARIMADataset):
    
    #splitting the dataset into training and testing set.
    seasonalARIMATrainingSet = seasonalARIMADataset.iloc[:717]
    seasonalARIMATestingSet = seasonalARIMADataset.iloc[717:]
    
    return seasonalARIMATrainingSet, seasonalARIMATestingSet

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readSeasonalARIMAXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readSeasonalARIMAXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save SeasonalARIMA as a pickle file.
"""
def saveSeasonalARIMAModel(seasonalARIMAModel):
    
    #Write SeasonalARIMAModel as a picke file
    with open("SeasonalARIMAModel.pkl",'wb') as seasonalARIMAModel_Pickle:
        pickle.dump(seasonalARIMAModel, seasonalARIMAModel_Pickle, protocol = 2)

"""
read SeasonalARIMA from pickle file
"""
def readSeasonalARIMAModel():
    
    #load SeasonalARIMAModel model
    with open("SeasonalARIMAModel.pkl","rb") as seasonalARIMAModel:
        seasonalARIMAModel = pickle.load(seasonalARIMAModel)
    
    return seasonalARIMAModel

"""
Save SeasonalARIMA as a pickle file.
"""
def saveSeasonalARIMAModelForFullDataset(seasonalARIMAModelForFullDataset):
    
    #Write SeasonalARIMAModelForFullDataset as a picke file
    with open("SeasonalARIMAModelForFullDataset.pkl",'wb') as seasonalARIMAModelForFullDataset_Pickle:
        pickle.dump(seasonalARIMAModelForFullDataset, seasonalARIMAModelForFullDataset_Pickle, protocol = 2)

"""
read SeasonalARIMA from pickle file
"""
def readSeasonalARIMAModelForFullDataset():
    
    #load SeasonalARIMAModelForFullDataset model
    with open("SeasonalARIMAModelForFullDataset.pkl","rb") as seasonalARIMAModelForFullDataset:
        seasonalARIMAModelForFullDataset = pickle.load(seasonalARIMAModelForFullDataset)
    
    return seasonalARIMAModelForFullDataset

"""
save SeasonalARIMA PredictedValues as a pickle file
"""

def saveSeasonalARIMAPredictedValues(seasonalARIMAPredictedValues):
    
    #Write SeasonalARIMAPredictedValues in a picke file
    with open("SeasonalARIMAPredictedValues.pkl",'wb') as seasonalARIMAPredictedValues_Pickle:
        pickle.dump(seasonalARIMAPredictedValues, seasonalARIMAPredictedValues_Pickle, protocol = 2)

"""
read SeasonalARIMA PredictedValues from pickle file
"""
def readSeasonalARIMAPredictedValues():
    
    #load SeasonalARIMAPredictedValues
    with open("SeasonalARIMAPredictedValues.pkl","rb") as seasonalARIMAPredictedValues_pickle:
        seasonalARIMAPredictedValues = pickle.load(seasonalARIMAPredictedValues_pickle)
    
    return seasonalARIMAPredictedValues

"""
save SeasonalARIMA ForecastedValues as a pickle file
"""

def saveSeasonalARIMAForecastedValues(seasonalARIMAForecastedValues):
    
    #Write SeasonalARIMAForecastedValues in a picke file
    with open("SeasonalARIMAForecastedValues.pkl",'wb') as seasonalARIMAForecastedValues_Pickle:
        pickle.dump(seasonalARIMAForecastedValues, seasonalARIMAForecastedValues_Pickle, protocol = 2)

"""
read SeasonalARIMAForecastedValues from pickle file
"""
def readSeasonalARIMAForecastedValues():
    
    #load SeasonalARIMAForecastedValues
    with open("SeasonalARIMAForecastedValues.pkl","rb") as seasonalARIMAForecastedValues_pickle:
        seasonalARIMAForecastedValues = pickle.load(seasonalARIMAForecastedValues_pickle)
    
    return seasonalARIMAForecastedValues


