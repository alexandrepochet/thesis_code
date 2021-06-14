import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pdb
import warnings



def main(): 
     """
     Execute matching action for testing
     """
     warnings.filterwarnings("ignore")
     start = time.time()
     location = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/output/"

     arima_daily = np.loadtxt(location + str('Arima_daily.txt'))
     arima_hourly = np.loadtxt(location + str('Arima_hourly.txt'))
     svc_daily = np.loadtxt(location + str('SVC_daily.txt'))
     svc_hourly = np.loadtxt(location + str('SVC_hourly.txt'))
     miningsvc_daily = np.loadtxt(location + str('MiningSVC_daily.txt'))
     miningsvc_hourly = np.loadtxt(location + str('MiningSVC_hourly.txt'))
     lstm_LOCF_daily = np.loadtxt(location + str('lstm_daily_LOCF.txt'))
     lstm_NA_daily = np.loadtxt(location + str('lstm_daily_NA.txt'))
     lstm_LOCF_hourly = np.loadtxt(location + str('lstm_hourly_LOCF.txt'))
     lstm_NA_hourly = np.loadtxt(location + str('lstm_hourly_NA.txt'))
     length_daily = len(arima_daily)
     length_hourly = len(arima_hourly)
     data_daily = [[np.count_nonzero(np.diff(arima_daily))/length_daily, 'ARMA', 'daily'],
           [np.count_nonzero(np.diff(svc_daily))/length_daily, 'SVC_TA', 'daily'],
           [np.count_nonzero(np.diff(miningsvc_daily))/length_daily, 'SVM_TM', 'daily'],
           [np.count_nonzero(np.diff(lstm_LOCF_daily))/length_daily, 'RNN_SA_LOCF', 'daily'],
           [np.count_nonzero(np.diff(lstm_NA_daily))/length_daily, 'RNN_SA_NA', 'daily'] ]
     data_hourly = [[np.count_nonzero(np.diff(arima_hourly))/length_hourly, 'ARMA', 'hourly'],
           [np.count_nonzero(np.diff(svc_hourly))/length_hourly, 'SVC_TA', 'hourly'],
           [np.count_nonzero(np.diff(miningsvc_hourly))/length_hourly, 'SVM_TM', 'hourly'],
           [np.count_nonzero(np.diff(lstm_LOCF_hourly))/length_hourly, 'RNN_SA_LOCF', 'hourly'],
           [np.count_nonzero(np.diff(lstm_NA_hourly))/length_hourly, 'RNN_SA_NA', 'hourly']]
     df_daily = pd.DataFrame(data_daily, columns = ['Turnover', 'Model', 'Frequency'])
     df_hourly = pd.DataFrame(data_hourly, columns = ['Turnover', 'Model', 'Frequency'])
     ax_daily = sns.catplot(x="Model", y="Turnover", kind="bar", data=df_daily)
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'Turnover_daily' + '.jpg', pad_inches=1)
     ax_hourly= sns.catplot(x="Model", y="Turnover", kind="bar", data=df_hourly)
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     plt.savefig(location + 'Turnover_hourly' + '.jpg', pad_inches=1)
     

if __name__ == '__main__':
    main()