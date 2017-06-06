# coding: utf-8

import os
import sys
import re
import time
from pyspark import SparkConf, SparkContext     # 重要
from pyspark.sql import SQLContext              # 
import pandas as pd

reload(sys)
sys.setdefaultencoding("utf-8")

# 0 add the envs                                # part0 重要
if "SPARK_HOME" not in os.environ:
    os.environ["SPARK_HOME"] = '/usr/hdp/2.3.4.0-3485/spark'

SPARK_HOME = os.environ["SPARK_HOME"]

sys.path.insert(0, os.path.join(SPARK_HOME, "python", "lib"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))

# 2 process each parquet

if __name__ == "__main__":
    start_time = time.time()
    try:
        sc.stop()                       # --------重要！否则可能下面定义sc会报错----------------
    except:
        pass
    conf = SparkConf().setAppName("UserLabelGenerate")
    #set("spark.driver.allowMultipleContexts", "true")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    rdd = sc.textFile("test/total_summary9")
    rdd.take(3)
    print rdd.count()


    #sc.stop()  
