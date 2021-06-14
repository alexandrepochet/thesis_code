import initial_analysis.TSA as t
import time
import data.data_currency as d
import pdb
import warnings
import numpy as np


def main(): 
     """
     Execute matching action for testing
     """
     nb_processes = 4
     warnings.filterwarnings("ignore")
     start = time.time()
     
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
     data = d.data_currency(file)
     # Daily
   
     print("resampling...\n")
     data.resample('D')
     threshold = 0#np.mean(data.df.close_bid_ask)*5
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = data.get_length()
     print(size)

     data_sliced = data.slice(0,int(0.8*size))
     model = t.TSA(data_sliced, threshold)
     print ('training the arx model...\n')
     train =[50, 100]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arx1 = model.run("time_series_analysis", nb_processes, train, test,
                                True, False, type_='initial')

     train =[250, 500, 1000, 2000]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arx2 = model.run("time_series_analysis", nb_processes, train, test,
                                True, False, type_='initial')
     results_D_Arx = results_D_Arx1 + results_D_Arx2
     # Daily Arima
     print ('training the arima model...\n')
     train =[50, 100]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arima1 = model.run("time_series_analysis", nb_processes, train, test,
                                   False, False, type_='initial')
     train =[250, 500, 1000, 2000]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arima2 = model.run("time_series_analysis", nb_processes, train, test,
                                  False, False, type_='initial')
     results_D_Arima = results_D_Arima1 + results_D_Arima2
     # Hourly ARX
     data = d.data_currency(file)
     print("resampling...\n")
     data.resample('H')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = data.get_length()
     data_sliced = data.slice(0,int(0.8*size))
     model = t.TSA(data_sliced, threshold)
     print ('training the arx model...\n')
     train =[250, 1000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arx1 = model.run("time_series_analysis", nb_processes, train, test,
                                 True, False, type_='initial')
     train =[5000, 10000, 20000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arx2 = model.run("time_series_analysis", nb_processes, train, test, 
                                True, False, type_='initial')
     results_H_Arx = results_H_Arx1 + results_H_Arx2

     # Hourly Arima
     print ('training the arima model...\n')
     train =[250, 1000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arima1 = model.run("time_series_analysis", nb_processes, train, test, 
                                  False, False, type_='initial')
     train =[5000, 10000, 20000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arima2 = model.run("time_series_analysis", nb_processes, train, test, 
                                  False, False, type_='initial')
     results_H_Arima = results_H_Arima1 + results_H_Arima2

     #printing
     print ('Results daily ARX...\n')
     print (results_D_Arx)
     print ('Results daily Arima...\n')
     print (results_D_Arima)
     
     print ('Results hourly ARX...\n')
     print (results_H_Arx)
     print ('Results hourly Arima...\n')
     print (results_H_Arima)     
     
     # test based on best training-testing windows
    
     #Daily
     data = d.data_currency(file)
     print("resampling...\n")
     data.resample('D')
     threshold = 0.0000
     data.define_threshold(threshold)
     size = data.get_length()
     print(size)

     data_sliced = data.slice(int(0.8*size - 100),size)
     model = t.TSA(data_sliced, threshold)
         
     print ('training the arx model...\n')
     train = 100
     test = 25
     results = model.time_series_analysis(train, test, True, False)
     model.plot_return(title = 'Daily_ARX')
     print('results daily ARX ' + str(results))
     
     
     # test based on best training-testing windows
     data_sliced = data.slice(int(0.8*size - 500),size)
     model = t.TSA(data_sliced, threshold)

     print ('training the Arima model...\n')
     train = 500
     test = 500
     results = model.time_series_analysis(train, test, False, False)
     model.plot_return(title = 'Daily_Arima')
     print('results daily Arima ' + str(results))
     
     # Hourly
     data = d.data_currency(file)
     print("resampling...\n")
     data.resample('H')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = data.get_length()
     print(size)
     
     # test based on best training-testing windows
     data_sliced = data.slice(int(0.8*size - 20000),size)
     model = t.TSA(data_sliced, threshold)

     print ('training the arx model...\n')
     train = 20000
     test = 20000
     results = model.time_series_analysis(train, test, True, False)
     model.plot_return('Hourly_ARX2', True)
     print('results hourly ARX ' + str(results))

     print ('training the Arima model...\n')
     train = 20000
     test = 20000
     results = model.time_series_analysis(train, test, False, False)
     model.plot_return('Hourly_Arima2', True)
     print('results hourly Arima ' + str(results)) 
     
     end = time.time()
     print(end - start)
     
if __name__ == '__main__':
    main()
