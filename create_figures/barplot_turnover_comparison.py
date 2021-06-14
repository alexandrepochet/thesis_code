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

     lstm_LOCF_daily = np.loadtxt(location + str('lstm_daily_LOCF.txt'))
     lstm_NA_daily = np.loadtxt(location + str('lstm_daily_NA.txt'))
     lstm_LOCF_hourly = np.loadtxt(location + str('lstm_hourly_LOCF.txt'))
     lstm_NA_hourly = np.loadtxt(location + str('lstm_hourly_NA.txt'))

     lstm_LOCF_return_daily = np.loadtxt(location + str('RNN_SA_LOCF_return_daily.txt'))
     lstm_NA_return_daily = np.loadtxt(location + str('RNN_SA_NA_return_daily.txt'))
     lstm_LOCF_return_hourly = np.loadtxt(location + str('RNN_SA_LOCF_return_hourly.txt'))
     lstm_NA_return_hourly = np.loadtxt(location + str('RNN_SA_NA_return_hourly.txt'))

     length_daily = len(lstm_LOCF_daily)
     length_hourly = len(lstm_LOCF_hourly)
     data_daily = [[np.count_nonzero(np.diff(lstm_LOCF_daily))/length_daily, 'RNN_SA_LOCF', 'Binary Cross-Entropy'],
           [np.count_nonzero(np.diff(lstm_NA_daily))/length_daily, 'RNN_SA_NA', 'Binary Cross-Entropy'],
            [np.count_nonzero(np.diff(lstm_LOCF_return_daily))/length_daily, 'RNN_SA_LOCF', 'Cost-Adjusted Return'],
           [np.count_nonzero(np.diff(lstm_NA_return_daily))/length_daily, 'RNN_SA_NA', 'Cost-Adjusted Return']]
     data_hourly = [[np.count_nonzero(np.diff(lstm_LOCF_hourly))/length_hourly, 'RNN_SA_LOCF', 'Binary Cross-Entropy'],
           [np.count_nonzero(np.diff(lstm_NA_hourly))/length_hourly, 'RNN_SA_NA', 'Binary Cross-Entropy'],
            [np.count_nonzero(np.diff(lstm_LOCF_return_hourly))/length_hourly, 'RNN_SA_LOCF', 'Cost-Adjusted Return'],
           [np.count_nonzero(np.diff(lstm_NA_return_hourly))/length_hourly, 'RNN_SA_NA', 'Cost-Adjusted Return']]
     df_daily = pd.DataFrame(data_daily, columns = ['Turnover', 'Model', 'Loss'])
     df_hourly = pd.DataFrame(data_hourly, columns = ['Turnover', 'Model', 'Loss'])
     ax_daily = sns.catplot(x="Model", y="Turnover", hue="Loss", kind="bar", data=df_daily, legend_out=True)
     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
     #ax_daily.despine(left=True)
     #ax_daily.set_ylabels("Turnover")
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'Turnover_daily_comparison' + '.jpg', pad_inches=1)
     ax_hourly= sns.catplot(x="Model", y="Turnover", hue="Loss", kind="bar", data=df_hourly , legend_out=True)
     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     plt.savefig(location + 'Turnover_hourly_comparison' + '.jpg', pad_inches=1)
     

if __name__ == '__main__':
    main()