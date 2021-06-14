import pandas as pd
import time
import pdb
import numpy as np
from utils.callback import callback


def main(): 
    """
    Execute matching action for testing
    """
    start = time.time()
    training =   [0.52,0.64,0.65,0.64,0.66,0.68,0.76,0.76,0.76,0.76,0.76,0.54,0.55,0.54,0.56,0.58,0.54,0.55,0.54,0.56,0.58]
    validation = [0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,0.41,00.51,0.54,0.56,0.58,0.54,0.55,0.54,0.56,0.58]
    model = callback()
    epoch = model.EarlyStopping(training, validation)
    print("Stopped epoch: " + str(epoch))
    end = time.time()
    print(end - start)
    
if __name__ == '__main__':
    main()
