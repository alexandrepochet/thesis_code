# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:43:03 2019

@author: alexa
Load, preprocess and represents the currency data on a chart
"""

import pandas as pd
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname as up
import matplotlib.dates as mdates
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime, date, time, timedelta
import pdb


class Currency(object): 
    
    """
    
    Represents the currency data. Preprocess, plot and resample the currency data.
    The direction of the market is derived from a given threshold and is classified
    as up, down or stable. The data are stored in formatted files, one for ask 
    and one for bid data. Alternatively, unformatted bid and ask files can be 
    passed as arguments.
    
    Attributes:
        fname_ask: Formatted file containing the ask data 
        fname_bid: Formatted file containing the bid data
        raw_file_ask: Unformatted file containing the ask data
        raw_file_bid: Unformatted file containing the bid data
        
    """
    
    def __init__(self, file = None, fname_ask = None, fname_bid = None, 
                 raw_file_ask = None, raw_file_bid = None):
            
        if raw_file_ask != None and raw_file_bid != None:
            data_ask, data_bid = self.__read_file(raw_file_ask, raw_file_bid)
            self.df = self.__preprocess(data_ask, data_bid)
        elif fname_ask != None and fname_bid != None:
            data_ask, data_bid = self.__open(fname_ask, fname_bid)
            self.df = self.__preprocess(data_ask, data_bid)
        else:
            self.df = self.__open__(file)
        self.df_resampled = self.df.copy()
    
    
    def resample(self, time_frame):
        
        """
        
        Resample the data based on the given time frame
        
        Args:
            time_frame: The time frame at which to resample
            
        """  
        
        def return_calc(row):
            if row.date_previous == row.date_lag:
                return np.log(row.close/row.close_lag)
            else:
                return np.NaN
        
        
        self.df_resampled = self.df.copy()
        # Resample given the time frame parameter
        self.df_resampled = self.df.resample(time_frame).agg(
                            OrderedDict([
                                        ('open', 'first'),
                                        ('high', 'max'),
                                        ('low', 'min'),
                                        ('close', 'last'),
                                        ('volume', 'sum'),
                                        ('open_bid_ask', 'first'),
                                        ('close_bid_ask', 'last'),
                                        ])
                            )
        self.df_resampled.replace(["NaN", 'NaT'], np.nan, inplace = True)
        self.df_resampled.dropna(how='any', inplace=True)
        self.df_resampled['Return'] = np.log(self.df_resampled.close).diff()
        self.df_resampled.dropna(how='any', inplace=True)
        
    
    def define_threshold(self, threshold):
        
        """
        
        Defines the direction of the market for each date
        based on the given threshold. If the return is higher
        than the threshold, direction is flagged as up, if the
        return is lower than -threshold, direction is flagged 
        as down, and flagged as stable if the return lies in
        between
        
        Args:
            threshold: The threshold
            
        """  
        
        def func(row, threshold):
            if row['Return'] > threshold:
                return 'up'
            elif row['Return'] < -threshold:
                return 'down' 
            else:
                return 'stable'
        
        self.df_resampled['Direction']  = self.df_resampled.apply(lambda x: func(x, threshold), axis=1)
        # Printing counts
        print(self.df_resampled['Direction'].value_counts())
    
    
    def data_analysis(self, resampled = None):
        
        """
        
        Plots several charts and save them. Plots the original 
        data by default. If resampled, plots the resampeld data
        
        Args:
            resampled: Boolean, if True, plot the resampled data.
                       
        """
        
        if resampled == True:
            data = self.df_resampled.copy()
        else:
            data = self.df.copy()
        years = mdates.YearLocator()  # every year
        yearsFmt = mdates.DateFormatter('%Y-%m-%d')
        
        # Plots (based on mid)
        fig, ax = plt.subplots()
        ax.plot(data.index, data.close)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.xticks(rotation=45)
        ax.grid(which='both', axis='x')
        plt.ylabel('Rate ($)')
        plt.title('EUR/USD exchange rate')  
        plt.savefig('./figures/rate.jpg', bbox_inches='tight', pad_inches=1)
        fig, ax = plt.subplots()
        ax.plot(data.index, data.volume)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.xticks(rotation=45)
        ax.grid(which='both', axis='x')
        plt.ylabel('Volume ($mln)')
        plt.title('Volume')  
        plt.savefig('./figures/volume.jpg', bbox_inches='tight', pad_inches=1)
        # Plots (based on mid)
        fig, ax = plt.subplots()
        ax.plot(data.index, data.Return*100)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.xticks(rotation=45)
        ax.grid(which='both', axis='x')
        plt.ylabel('Return (%)')
        plt.title('Return of EUR/USD exchange rate')  
        plt.savefig('./figures/Return.jpg', bbox_inches='tight', pad_inches=1)
        fig, ax = plt.subplots()
        ax.plot(data.index, data.open_bid_ask)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.xticks(rotation=45)
        ax.grid(which='both', axis='x')
        plt.ylabel('bid-ask spread ($)')
        plt.title('bid-ask spread of EUR/USD exchange rate')  
        plt.savefig('./figures/bid_ask.jpg', bbox_inches='tight', pad_inches=1)
    
    
    def __open(self, fname_ask, fname_bid):
        
        """
        
        Open the files specified as arguments
        
        Args:
            fname_ask: A file containing the ask data
            fname_bid: A file containing the bid data
            
        Returns:
            Two dataframes with the bid and ask data
            
        """  
          
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') 
        # Open file ask
        data_ask = pd.read_csv(fname_ask, index_col=0, date_parser=mydateparser, sep = "\t")
        # Open file bid
        data_bid = pd.read_csv(fname_bid, index_col=0, date_parser=mydateparser, sep = "\t")
        
        return data_ask, data_bid
    
    
    def __open__(self, file):
        
        """
        
        Open the file specified as argument
        
        Args:
            file: A file containing the preprocessed data
            
        Returns:
            A dataframe with the data
            
        """  
          
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') 
        # Open file 
        data = pd.read_csv(file, index_col=0, date_parser=mydateparser,
                           sep = "\t")
        
        return data
    
    
    def __preprocess(self, data_ask, data_bid):
        
        """
        
        Preprocess the currency data
        
        Args:
            data_ask: A dataframe with the ask data
            data_bid: A dataframe with the bid data
            
        Returns:
            A dataframe with the processed data
            
        """
        def return_calc(row):
            if row.date_previous == row.date_lag:
                return np.log(row.close/row.close_lag)
            else:
                return np.NaN

        # Remove duplicates in time
        data_ask=data_ask.reset_index().drop_duplicates(subset='Date', keep='last').set_index('Date')
        # Description
        data_ask.dropna().describe()
        # Rename columns
        data_ask.rename(columns={'open': 'open_ask', 'high': 'high_ask', 'low': 'low_ask',
                                 'close': 'close_ask', 'volume': 'volume_ask'}, inplace=True)
        # Remove duplicates in time
        data_bid=data_bid.reset_index().drop_duplicates(subset='Date', keep='last').set_index('Date')
        # Description
        data_bid.dropna().describe()
        # Rename columns
        data_bid.rename(columns={'open': 'open_bid', 'high': 'high_bid', 'low': 'low_bid',
                                 'close': 'close_bid', 'volume': 'volume_bid'}, inplace=True)
        # Join both database
        data = data_ask.merge(data_bid, on = 'Date')
        # Calculate mid
        data['open_mid'] = (data['open_ask'] + data['open_bid'])/2
        data['high_mid'] = (data['high_ask'] + data['high_bid'])/2
        data['low_mid'] = (data['low_ask'] + data['low_bid'])/2
        data['close_mid'] = (data['close_ask'] + data['close_bid'])/2
        data['volume_mid'] = (data['volume_ask'] + data['volume_bid'])/2     
        data['Return'] =np.log(data.close_mid).diff()
        data['open_bid_ask'] = data['open_ask'] - data['open_bid']
        data['close_bid_ask'] = data['close_ask'] - data['close_bid']
        data.dropna(how='any', inplace=True)
        columns = ['open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask',
                   'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid']
        data.drop(columns, inplace=True, axis=1)
        columns = ['index_x', 'index_y']
        try:
            data.drop(columns, inplace=True, axis=1)
        except:
            print('')
        data.rename(columns={'open_mid': 'open', 'high_mid': 'high',
                             'low_mid': 'low', 'close_mid': 'close',
                             'volume_mid': 'volume'}, inplace=True)   
        path = str(up(up(up(__file__)))) + "/forexDATA/"
        data.to_csv(str(path) + 'currency6.txt', header=True, 
                    index=True, sep='\t' , float_format='%.6f')
    
        return data
        
        
    def __read_file(self, raw_file_ask, raw_file_bid):
        
        """
        
        Read ask and bid files and store the data in two dataframes
        
        Args:
            raw_file_ask: the file with ask data
            raw_file_bid: the file with bid data 
            
        Returns:
            The ask and bid data stored in two dataframes
            
        """
        
        def fun(row, format_):
            return pd.datetime.strptime(row['Date'], format_)
        
        if raw_file_ask == None or raw_file_bid == None:
            print("no raw file provided")
        else:
            elements = defaultdict(list)
            with open(raw_file_ask, "r") as file:
                i = 0
                lines = file.read().split("\n")
                print (len(lines))
                for line in lines:
                    currentline = line.split(",")
                    if i != 0:
                        try:
                            elements['Date'].append(str(currentline[0]))
                            elements['open'].append(float(currentline[1]))
                            elements['high'].append(float(currentline[2]))
                            elements['low'].append(float(currentline[3]))
                            elements['close'].append(float(currentline[4]))
                            elements['volume'].append(float(currentline[5]))
                        except:
                            continue
                    i = i + 1
                    if i%10000==0:
                        print (i)
            
            file.close()       
            data_ask=pd.DataFrame({'Date': pd.Index(elements['Date']),
                                   'open': pd.Index(elements['open']),
                                   'high': pd.Index(elements['high']),
                                   'low': pd.Index(elements['low']),
                                   'close': pd.Index(elements['close']),
                                   'volume': pd.Index(elements['volume'])})
            data_ask['Date'] = data_ask.apply(lambda x: fun(x, '%d.%m.%Y %H:%M:%S.%f'), axis=1)
            
            elements = defaultdict(list)
            with open(raw_file_bid, "r") as file:
                i = 0
                lines = file.read().split("\n")
                print (len(lines))
                for line in lines:
                    currentline = line.split(",")
                    if i != 0:
                        try:
                            elements['Date'].append(str(currentline[0]))
                            elements['open'].append(float(currentline[1]))
                            elements['high'].append(float(currentline[2]))
                            elements['low'].append(float(currentline[3]))
                            elements['close'].append(float(currentline[4]))
                            elements['volume'].append(float(currentline[5]))
                        except:
                            continue
                    i = i + 1
                    if i%10000==0:
                        print (i)
            
            file.close()
            data_bid=pd.DataFrame({'Date': pd.Index(elements['Date']),
                                   'open': pd.Index(elements['open']),
                                   'high': pd.Index(elements['high']),
                                   'low': pd.Index(elements['low']),
                                   'close': pd.Index(elements['close']),
                                   'volume': pd.Index(elements['volume'])})
            data_bid['Date'] = data_bid.apply(lambda x: fun(x, '%d.%m.%Y %H:%M:%S.%f'), axis=1)
            
            path = str(up(up(up(__file__)))) + "/forexDATA/"
            data_ask.to_csv(str(path) + 'eurusd_ask_full.csv', sep='\t', index = False)
            data_bid.to_csv(str(path) + 'eurusd_bid_full.csv', sep='\t', index = False)
            
            return data_ask, data_bid
            
    
class Price(object):
    
    def __init__(self, open_, high, low, close, volume, bid_ask):
            self.open = open_
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume
            self.bid_ask = bid_ask


    def get_open_ask(self):
        return self.open + self.bid_ask/2    

    
    def get_open_bid(self):
        return self.open - self.bid_ask/2
    
    
    def get_close_ask(self):
        return self.close + self.bid_ask/2
    
    
    def get_close_bid(self):
        return self.close - self.bid_ask/2

    