from collections import defaultdict, OrderedDict
from os.path import dirname as up
from matplotlib import pyplot as plt, dates as mdates
import numpy as np
import pandas as pd
import pdb

class currency_preprocess():
    """
    Read currency file and format it appropriately for further treatment
    Saves the output into a csv file
    """
    def format_file(self, raw_file_ask, raw_file_bid):
        data_ask, data_bid = self._read_file(raw_file_ask, raw_file_bid)
        return data_ask, data_bid

    def _read_file(self, raw_file_ask, raw_file_bid):
        """
        Read ask and bid files and store the data in two dataframes

        Returns:
            The ask and bid data stored in two dataframes
        """
        def fun(row, format_):
            return pd.datetime.strptime(row['Date'], format_)

        if raw_file_ask is None or raw_file_bid is None:
            print("no raw file provided")
        else:
            elements = defaultdict(list)
            with open(raw_file_ask, "r") as file:
                i = 0
                lines = file.read().split("\n")
                print(len(lines))
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
                            print('issue ' + str(i) + ' ask \n')
                            continue
                    i = i + 1
                    if i%10000 == 0:
                        print(i)

            file.close()
            data_ask = pd.DataFrame({'Date': pd.Index(elements['Date']),
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
                print(len(lines))
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
                            print('issue ' + str(i) + ' bid \n')
                            continue
                    i = i + 1
                    if i%10000 == 0:
                        print(i)

            file.close()
            data_bid = pd.DataFrame({'Date': pd.Index(elements['Date']),
                                     'open': pd.Index(elements['open']),
                                     'high': pd.Index(elements['high']),
                                     'low': pd.Index(elements['low']),
                                     'close': pd.Index(elements['close']),
                                     'volume': pd.Index(elements['volume'])})
            data_bid['Date'] = data_bid.apply(lambda x: fun(x, '%d.%m.%Y %H:%M:%S.%f'), axis=1)

            path = str(up(up(up(__file__)))) + "/forexDATA/"
            data_ask.to_csv(str(path) + 'eurusd_ask_full.csv', sep='\t', index=False)
            data_bid.to_csv(str(path) + 'eurusd_bid_full.csv', sep='\t', index=False)
            return data_ask, data_bid

    def preprocess(self, data_ask, data_bid, title):
        """
        Preprocess the currency data

        Args:
            data_ask: A dataframe with the ask data
            data_bid: A dataframe with the bid data
        Returns:
    
        """
        # Remove duplicates in time
        data_ask = data_ask.reset_index().drop_duplicates( \
                       subset='Date', keep='last').set_index('Date')
        # Description
        data_ask.dropna().describe()
        # Rename columns
        data_ask.rename(columns={ \
            'open': 'open_ask', 'high': 'high_ask', 'low': 'low_ask',
            'close': 'close_ask', 'volume': 'volume_ask'}, inplace=True)
        # Remove duplicates in time
        data_bid = data_bid.reset_index().drop_duplicates( \
                       subset='Date', keep='last').set_index('Date')
        # Description
        data_bid.dropna().describe()
        # Rename columns
        data_bid.rename(columns={'open': 'open_bid', 'high': 'high_bid', 'low': 'low_bid',
                                 'close': 'close_bid', 'volume': 'volume_bid'}, inplace=True)
        # Join both database
        data = data_ask.merge(data_bid, on='Date')
        # Calculate mid
        data['open_mid'] = (data['open_ask'] + data['open_bid'])/2
        data['high_mid'] = (data['high_ask'] + data['high_bid'])/2
        data['low_mid'] = (data['low_ask'] + data['low_bid'])/2
        data['close_mid'] = (data['close_ask'] + data['close_bid'])/2
        data['volume_mid'] = (data['volume_ask'] + data['volume_bid'])/2
        data['Return'] = np.log(data.close_mid).diff()
        data['open_bid_ask'] = data['open_ask'] - data['open_bid']
        data['close_bid_ask'] = data['close_ask'] - data['close_bid']
        data['close_bid_ask_previous'] = data['close_bid_ask'].shift(1)
        data.dropna(how='any', inplace=True)
        columns = ['open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask',
                   'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid']
        data.drop(columns, inplace=True, axis=1)
        columns = ['index_x', 'index_y']
        try:
            data.drop(columns, inplace=True, axis=1)
        except:
            print('issue \n')
        data.rename(columns={'open_mid': 'open', 'high_mid': 'high',
                             'low_mid': 'low', 'close_mid': 'close',
                             'volume_mid': 'volume'}, inplace=True)
        path = str(up(up(up(__file__)))) + "/forexDATA/"
        data.to_csv(str(path) + title +'.txt', header=True,
                    index=True, sep='\t', float_format='%.6f')
        return data


class currency_analysis():

    def data_analysis(self, data):
     
        years = mdates.YearLocator()  # every year
        years_fmt = mdates.DateFormatter('%Y-%m-%d')

        # Plots (based on mid)
        fig, axis = plt.subplots()
        axis.plot(data.index, data.close)
        # format the ticks
        axis.xaxis.set_major_locator(years)
        axis.xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        axis.grid(which='both', axis='x')
        plt.ylabel('Rate ($)')
        plt.title('EUR/USD exchange rate')
        plt.savefig('./figures/rate2.jpg', bbox_inches='tight', pad_inches=1)
        fig, axis = plt.subplots()
        axis.plot(data.index, data.volume)
        # format the ticks
        axis.xaxis.set_major_locator(years)
        axis.xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        axis.grid(which='both', axis='x')
        plt.ylabel('Volume ($mln)')
        plt.title('Volume')
        plt.savefig('./figures/volume2.jpg', bbox_inches='tight', pad_inches=1)
        # Plots (based on mid)
        fig, axis = plt.subplots()
        axis.plot(data.index, data.Return*100)
        # format the ticks
        axis.xaxis.set_major_locator(years)
        axis.xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        axis.grid(which='both', axis='x')
        plt.ylabel('Return (%)')
        plt.title('Return of EUR/USD exchange rate')
        plt.savefig('./figures/Return2.jpg', bbox_inches='tight', pad_inches=1)
        fig, axis = plt.subplots()
        axis.plot(data.index, data.open_bid_ask)
        # format the ticks
        axis.xaxis.set_major_locator(years)
        axis.xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        axis.grid(which='both', axis='x')
        plt.ylabel('bid-ask spread ($)')
        plt.title('bid-ask spread of EUR/USD exchange rate')
        plt.savefig('./figures/bid_ask2.jpg', bbox_inches='tight', pad_inches=1)


class currency():

    def resample(self, df, time_frame):
        # Resample given the time frame parameter
        df = df.resample(time_frame).agg( \
                        OrderedDict([('open', 'first'),
                                    ('high', 'max'),
                                    ('low', 'min'),
                                    ('close', 'last'),
                                    ('volume', 'sum'),
                                    ('open_bid_ask', 'first'),
                                    ('close_bid_ask', 'last'),
                                    ('close_bid_ask_previous', 'last'),
                                    ])
                        )
        df.replace(["NaN", 'NaT'], np.nan, inplace=True)
        df.dropna(how='any', inplace=True)
        df['Return'] = np.log(df.close).diff()
        df.dropna(how='any', inplace=True)
        return df

    def define_threshold(self, df, threshold):
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

        df['Direction'] = df.apply(lambda x: func(x, threshold), axis=1)
        # Printing counts
        print(df['Direction'].value_counts())
        return df
