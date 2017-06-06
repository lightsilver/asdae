"""
version of for loops
"""
import os
import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
import time
import pandas as pd
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")

# 0 add the envs
if "SPARK_HOME" not in os.environ:
        os.environ["SPARK_HOME"] = '/usr/hdp/2.3.4.0-3485/spark'

SPARK_HOME = os.environ["SPARK_HOME"]

sys.path.insert(0, os.path.join(SPARK_HOME, "python", "lib"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))

try:
        sc.stop()
except:
        pass
conf = SparkConf().setAppName("ReadUsersData")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
rdd = MLUtils.loadLibSVMFile(sc,"/sourcedata/zhaih/s1")

n = 20000			# number of records to be processed
d = 7799
M = np.zeros((n,d)).astype(np.int8)
res = rdd.take(n)
begin_time = time.time()

for i in range(n):
    #id = res[i].label
    values = res[i].features
    for j in range(d):
	    M[i][j] = int(values[j])

# def sparse2dense(res):
#     import numpy as np
#
#     #id = res[i].label
#     d = 7799
#     record = np.zeros((1,d)).astype(np.int8)
#     values = res.features
#     for j in range(d):
# 	record[0][j] = int(values[j])
#     return record
    
# M2 = rdd.map(sparse2dense).collect()
print time.time()-begin_time
M = M.astype(np.int8)
print M
np.save('./users_data.npy',M)
