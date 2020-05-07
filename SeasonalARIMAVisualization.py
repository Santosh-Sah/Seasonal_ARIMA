# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def visualizeSeasonalARIMAPredictedValues(seasonalARIMAXTest, 
                                          seasonalARIMAPredictedValues):
    
    #plotting the predicted values, and testing set
    title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
    
    ylabel='parts per million'
    
    xlabel='' 

    ax = seasonalARIMAXTest['interpolated'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAPredictedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('PredeictedValues.png')

def visualizeSeasonalARIMAForecastedValues(seasonalARIMADataset, 
                                           seasonalARIMAForecastedValues):
    
    #plotting the predicted values, and testing set
    title = 'Real Manufacturing and Trade Inventories'
    
    ylabel='parts per million'
    
    xlabel='' 

    ax = seasonalARIMADataset['interpolated'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAForecastedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('ForecastedValues.png')

def visualizeACFPlot(seasonalARIMADataset):
    
    title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
    lags = 40
    plot_acf(seasonalARIMADataset['interpolated'],title=title,lags=lags)
    pylab.savefig('acf_plot.png')

def visualizePACFPlot(seasonalARIMADataset):
    
    title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
    lags = 40
    plot_pacf(seasonalARIMADataset['interpolated'],title=title,lags=lags)
    pylab.savefig('pacf_plot.png')