import os
import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
import time
import pandas as pd
import numpy as np

#reload(sys)
#sys.setdefaultencoding("utf-8")


from libsvm.python.svmutil import *
from libsvm.python.svm import *

y, x = svm_read_problem('goods_vectors_20170605.libsvm')
n = 5448#10000			# number of records to be processed
d = 7799
M = np.zeros((n,d+1)).astype(np.int8)

for i in range(n):
    id = y[i]
    values = x[i]
    M[i][d] = int(id)
    for j in range(d):
        M[i][j] = int(values[j])



def sparse2dense(res):
    id = res.label
    d = 7799
    record = np.zeros((1,d+1)).astype(np.int8)
    values = res.features
    for j in range(d):
        record[0][j] = int(values[j])
    record[0][d] = int(id)
    return record
    
#M2 = rdd.map(sparse2dense).collect()
print(time.time()-begin_time)
M = M.astype(np.int8)
print(M)
np.save('./goods_data.npy',M)

