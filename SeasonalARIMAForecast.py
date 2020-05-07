# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:57 2020

@author: Santosh Sah
"""

from SeasonalARIMAUtils import (importSeasonalARIMADataset, 
                                saveSeasonalARIMAForecastedValues,
                                readSeasonalARIMAForecastedValues, 
                                readSeasonalARIMAModelForFullDataset)

from SeasonalARIMAVisualization import visualizeSeasonalARIMAForecastedValues

def forecastSeasonalARIMAModel():
    
    #reading the dataset
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    #reading the model whichis trained on the whole dataset
    seasonalARIMAModel = readSeasonalARIMAModelForFullDataset()
    
    #forecasting for 11 months
    seasonalARIMAForecastedValues = seasonalARIMAModel.predict(len(seasonalARIMADataset),
                                                               len(seasonalARIMADataset)+11,
                                                               typ='levels').rename("SARIMAX(0, 1, 1)x(2, 0, 1, 12) Prediction")
    
    #saving the forecasted values
    saveSeasonalARIMAForecastedValues(seasonalARIMAForecastedValues)

def plotSeasonalARIMAForecastedValues():
    
    #reading the dataset
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    #reading the forecated values
    seasonalARIMAForecastedValues = readSeasonalARIMAForecastedValues()
    
    #visualizing the forecated values
    visualizeSeasonalARIMAForecastedValues(seasonalARIMADataset, 
                                           seasonalARIMAForecastedValues)

if __name__ == "__main__":
    #forecastSeasonalARIMAModel()
    plotSeasonalARIMAForecastedValues()
    