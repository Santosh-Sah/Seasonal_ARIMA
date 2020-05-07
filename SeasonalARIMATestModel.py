# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""

from SeasonalARIMAUtils import (readSeasonalARIMAXTest, 
                                readSeasonalARIMAModel, 
                                saveSeasonalARIMAPredictedValues, 
                                readSeasonalARIMAXTrain,
                                readSeasonalARIMAPredictedValues)

from SeasonalARIMAVisualization import visualizeSeasonalARIMAPredictedValues

"""
test the model on testing dataset
"""
def testSeasonalARIMAModel():
    
    #reading the training dataset
    X_train = readSeasonalARIMAXTrain()
    
    #reading testing set
    X_test = readSeasonalARIMAXTest()
    
    start = len(X_train)
    
    end = len(X_train) + len(X_test) - 1
    
    #reading model from pickle file
    seasonalARIMAModel = readSeasonalARIMAModel()
    
    #forecasting
    #Passing dynamic=False means that forecasts at each point are generated using the full history up to that point (all lagged values).
    #Passing typ='levels' predicts the levels of the original endogenous variables. 
    #If we'd used the default typ='linear' we would have seen linear predictions in terms of the differenced endogenous variables.
    predictedValues = seasonalARIMAModel.predict(start = start, end = end, dynamic = False, typ = "levels").rename("SARIMAX(0, 1, 1)x(2, 0, 1, 12) Prediction")
    
    #saving the foreasted values
    saveSeasonalARIMAPredictedValues(predictedValues)

def plotSeasonalARIMAPredictedValues():
    
    #reading testing set
    X_test = readSeasonalARIMAXTest()
    
    #reading predicted value
    predictedValues = readSeasonalARIMAPredictedValues()
    
    #visualizing the predicted values with training set and the testing set
    visualizeSeasonalARIMAPredictedValues(X_test, predictedValues)
    
    
if __name__ == "__main__":
    #testSeasonalARIMAModel()
    plotSeasonalARIMAPredictedValues()