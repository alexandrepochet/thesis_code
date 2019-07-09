# -*- coding: utf-8 -*-
import math


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()
    

def calculate_lags(df, lags):

    """
        
    Calculate the lag variables of the return
        
    Args:
        df: Data
        lags: Number of lags to calculate 

    Returns
                
    """    
    
    for i in range(1, lags + 1):
        df['Return_lag_' + str(i)] = df.Return.shift(i)

    df = df.iloc[lags:]