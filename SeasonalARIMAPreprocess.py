# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from SeasonalARIMAUtils import (importSeasonalARIMADataset, saveTrainingAndTestingDataset, 
                                splitSeasonalARIMADataset)

def preprocess():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    X_train, X_test = splitSeasonalARIMADataset(seasonalARIMADataset)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    

if __name__ == "__main__":
    preprocess()