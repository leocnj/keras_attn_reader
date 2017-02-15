import os
import sys
import pickle
import numpy as np
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)
from data_generator import DataGenerator

path = os.path.join(module_home, "datasets/toy_dataset/cnn_processed/")
batch_size = 32
trainData = DataGenerator(batch_size, path, 'training')

counter = 0
X, y = trainData.next()

for i in xrange(10):
    print "--------------"
    X, y = trainData.next()
    if (X[0][0].shape != (1948,)):
        print "X[0][0] instead of (1948,)"
        print X[0][0].shape
        print "--"
    if (X[1][0].shape != (20,)):
        print "X[1][0] instead of (20,)"
        print X[1][0].shape
        print "--"
    if (y[0].shape != (367,)):
        print "y[0] instead of (367,)"
        print y[0].shape
        print "--"
    if (X[0].shape != (batch_size,1948)):
        print "X[0] instead of (3, 1948)"
        print X[0].shape
        print "--"
    if (X[1].shape != (batch_size,20)):
        print "X[1] instead of (3,20)"
        print X[1].shape
        print "--"
    if (y.shape != (batch_size, 367)):
        print "y instead of (3, 367)"
        print y.shape
        print "--"

completeData = DataGenerator(batch_size, path, 'validation', complete=True)

counter = 0
vectorData, completeData = completeData.next()
#TODO: Write test for complete data generation, for now just print out the output for complete data
#print vectorData
