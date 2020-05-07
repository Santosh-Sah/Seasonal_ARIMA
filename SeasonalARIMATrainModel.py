# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.statespace.tools import diff

from SeasonalARIMAUtils import (saveSeasonalARIMAModel, 
                                readSeasonalARIMAXTrain, 
                                importSeasonalARIMADataset, 
                                saveSeasonalARIMAModelForFullDataset,
                                agumentedDickeyFullerTest)

from SeasonalARIMAVisualization import visualizeACFPlot, visualizePACFPlot


"""
Train SeasonalARIMA model on training set
"""
def trainSeasonalARIMAModel():
    
    X_train = readSeasonalARIMAXTrain()
    
    X_train["interpolated"] = X_train["interpolated"].astype('float64')
    
    #training model on the training set
    seasonalARIMAModel = SARIMAX(X_train['interpolated'],order=(0,1,1),seasonal_order=(2,0,1,12))
    
    seasonalARIMAModelFitResult = seasonalARIMAModel.fit()
    
    saveSeasonalARIMAModel(seasonalARIMAModelFitResult)
    
    print(seasonalARIMAModelFitResult.summary())
    
# =============================================================================
#                                          SARIMAX Results
#     ==========================================================================================
#     Dep. Variable:                       interpolated   No. Observations:                  717
#     Model:             SARIMAX(0, 1, 1)x(2, 0, 1, 12)   Log Likelihood                -205.153
#     Date:                            Fri, 08 May 2020   AIC                            420.307
#     Time:                                    00:17:22   BIC                            443.175
#     Sample:                                03-01-1958   HQIC                           429.138
#                                          - 11-01-2017
#     Covariance Type:                              opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     ma.L1         -0.3710      0.034    -10.855      0.000      -0.438      -0.304
#     ar.S.L12       1.0000      0.041     24.612      0.000       0.920       1.080
#     ar.S.L24      -0.0004      0.041     -0.009      0.993      -0.080       0.079
#     ma.S.L12      -0.8637      0.027    -31.480      0.000      -0.917      -0.810
#     sigma2         0.0960      0.005     20.133      0.000       0.087       0.105
#     ===================================================================================
#     Ljung-Box (Q):                       51.93   Jarque-Bera (JB):                 3.73
#     Prob(Q):                              0.10   Prob(JB):                         0.15
#     Heteroskedasticity (H):               1.13   Skew:                            -0.02
#     Prob(H) (two-sided):                  0.36   Kurtosis:                         3.35
#     ===================================================================================
# =============================================================================
    
"""
Train SeasonalARIMA model on full dataset
"""
def trainSeasonalARIMAModelOnFullDataset():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    seasonalARIMADataset["interpolated"] = seasonalARIMADataset["interpolated"].astype('float64')
    
    #training model on the whole dataset
    seasonalARIMAModel = SARIMAX(seasonalARIMADataset['interpolated'],order=(0,1,1),seasonal_order=(2,0,1,12))
    
    seasonalARIMAModelFitResult = seasonalARIMAModel.fit()
    
    saveSeasonalARIMAModelForFullDataset(seasonalARIMAModelFitResult)
    
    print(seasonalARIMAModelFitResult.summary())
    
# =============================================================================
#                                         SARIMAX Results
#     ==========================================================================================
#     Dep. Variable:                       interpolated   No. Observations:                  729
#     Model:             SARIMAX(0, 1, 1)x(2, 0, 1, 12)   Log Likelihood                -209.445
#     Date:                            Fri, 08 May 2020   AIC                            428.890
#     Time:                                    00:18:32   BIC                            451.842
#     Sample:                                03-01-1958   HQIC                           437.746
#                                          - 11-01-2018
#     Covariance Type:                              opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     ma.L1         -0.3718      0.034    -10.800      0.000      -0.439      -0.304
#     ar.S.L12       0.9996      0.041     24.196      0.000       0.919       1.081
#     ar.S.L24    4.464e-05      0.041      0.001      0.999      -0.081       0.081
#     ma.S.L12      -0.8653      0.026    -33.124      0.000      -0.917      -0.814
#     sigma2         0.0963      0.005     20.225      0.000       0.087       0.106
#     ===================================================================================
#     Ljung-Box (Q):                       51.04   Jarque-Bera (JB):                 3.75
#     Prob(Q):                              0.11   Prob(JB):                         0.15
#     Heteroskedasticity (H):               1.13   Skew:                            -0.04
#     Prob(H) (two-sided):                  0.36   Kurtosis:                         3.34
#     ===================================================================================
# =============================================================================


def testIsDatasetStationary():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    #order of p,d,q and P, D, Q is SARIMAX(0, 1, 1)x(2, 0, [1, 2], 12)
    #hence we take the first difference as d is 1 to check stationarity.
    seasonalARIMADataset["diff1"] = diff(seasonalARIMADataset["interpolated"], k_diff = 1)
    
    agumentedDickeyFullerTest(seasonalARIMADataset["diff1"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic       -5.200976
#     p-value                   0.000009
#     # lags used              20.000000
#     # observations          707.000000
#     critical value (1%)      -3.439633
#     critical value (5%)      -2.865637
#     critical value (10%)     -2.568952
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
def determineSARIMAOrderOfPAndQ():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    
    # For SARIMA Orders we set seasonal=True and pass in an m value
    autoArimaResult = auto_arima(seasonalARIMADataset["interpolated"], seasonal = True, m = 12)
    
    print(autoArimaResult.summary()) #order SARIMAX(0, 1, 0) 
    
# =============================================================================
#                                             SARIMAX Results
#     ===============================================================================================
#     Dep. Variable:                                       y   No. Observations:                  729
#     Model:             SARIMAX(0, 1, 1)x(2, 0, [1, 2], 12)   Log Likelihood                -208.336
#     Date:                                 Fri, 08 May 2020   AIC                            430.673
#     Time:                                         00:03:43   BIC                            462.805
#     Sample:                                              0   HQIC                           443.071
#                                                      - 729
#     Covariance Type:                                   opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     intercept   7.965e-05      0.000      0.353      0.724      -0.000       0.001
#     ma.L1         -0.3732      0.030    -12.253      0.000      -0.433      -0.314
#     ar.S.L12       0.0317      0.029      1.085      0.278      -0.026       0.089
#     ar.S.L24       0.9677      0.029     33.010      0.000       0.910       1.025
#     ma.S.L12       0.1276      0.037      3.493      0.000       0.056       0.199
#     ma.S.L24      -0.8654      0.055    -15.849      0.000      -0.972      -0.758
#     sigma2         0.0952      0.006     16.289      0.000       0.084       0.107
#     ===================================================================================
#     Ljung-Box (Q):                       50.35   Jarque-Bera (JB):                 4.24
#     Prob(Q):                              0.13   Prob(JB):                         0.12
#     Heteroskedasticity (H):               1.12   Skew:                            -0.04
#     Prob(H) (two-sided):                  0.38   Kurtosis:                         3.37
#     ===================================================================================
# =============================================================================
    
def plotACFPlot():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    visualizeACFPlot(seasonalARIMADataset)

def plotPACFPlot():
    
    seasonalARIMADataset = importSeasonalARIMADataset("co2_mm_mlo.csv")
    visualizePACFPlot(seasonalARIMADataset)
        
if __name__ == "__main__":
    #determineSARIMAOrderOfPAndQ()
    #testIsDatasetStationary()   
    #plotACFPlot()
    #plotPACFPlot()
    #trainSeasonalARIMAModel()
    trainSeasonalARIMAModelOnFullDataset()
