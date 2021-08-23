# Load a part of the test dataset
from readers.nowcast_reader import read_data
import os

os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import numpy as np
sys.path.append('../src/')
x_test,y_test = read_data('../data/sample/nowcast_testing.h5',end=50)