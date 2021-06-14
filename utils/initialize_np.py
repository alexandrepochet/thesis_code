import numpy as np
import random

def initialize(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)