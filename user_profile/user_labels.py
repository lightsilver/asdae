# coding: utf-8

# 1 read files
import os
import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pandas as pd

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

conf = SparkConf().setAppName("TagUser")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# 1 read file
total_data = sc.pickleFile("/user/mjoys/test/*/*")
print total_data.count()
#  transfer to dataframe
total_data = total_data.toDF(["imei", "pkg", "cnt"])

result = total_data.groupBy("imei", "pkg").agg({"cnt":"sum"})
result.selectExpr("imei as imei", "pkg as pkg", "`sum(cnt)` as cnt")
print result.count()
print result.show()
# 2 create the rules
#age = {"news_article": 10}
#for k, v in age.iteritems():
#	if k in user_vectors.columns:
#		user_vectors['old'] = map(int, user_vectors[k] > v)
# 3 tag
#print(user_vectors)
