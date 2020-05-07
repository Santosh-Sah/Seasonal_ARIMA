# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from SeasonalARIMAUtils import (readSeasonalARIMAXTest, 
                                readSeasonalARIMAPredictedValues)

"""

calculating SeasonalARIMA metrics

"""
def testSeasonalARIMAMetrics():
    
    #reading testing set
    X_test = readSeasonalARIMAXTest()
    
    X_test = X_test[["interpolated"]]
    
    #reading predicted value
    predictedValues = readSeasonalARIMAPredictedValues()
    
    meanSquredError = mean_squared_error(X_test, predictedValues)
    
    meanAbsoluteError = mean_absolute_error(X_test, predictedValues)
    
    rootMeanSquaredError = np.sqrt(mean_squared_error(X_test, predictedValues))
    
    print(meanSquredError) #0.11996607440710379
    
    print(meanAbsoluteError) #0.2577633496802605
    
    print(rootMeanSquaredError) #0.34636119067687676
    
    
    
if __name__ == "__main__":
    testSeasonalARIMAMetrics()