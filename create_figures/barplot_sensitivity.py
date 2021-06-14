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

     data = [[1.1196616463792037, 'SW'],
           [1.0647823395915, 'SO'],
           [ 1.061275015473489, 'V'],
           [1.0410563234990717, 'POMS']]
     
     df = pd.DataFrame(data, columns = ['S', 'Features excluded'])
     ax_daily = sns.catplot(x="Features excluded", y="S", data=df, kind="bar")
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'S_daily' + '.jpg', pad_inches=1)

if __name__ == '__main__':
    main()