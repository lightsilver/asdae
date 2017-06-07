# coding: utf-8

import os
import sys
import re
import time
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


def process_concated_table(table):
	"""
	对聚合后的表进行聚合
	"""
	df_group = table.groupBy("imei", "pkg").agg({"cnt_freq": "sum"})
	df_group = df_group.selectExpr("imei as imei", "pkg as pkg", "`sum(cnt_freq)` as cnt_freq")
	return df_group

# 2 process each parquet

if __name__ == "__main__":
	start_time = time.time()
	try:
		sc.stop()
	except:
		pass
	conf = SparkConf().setAppName("UserLabelGenerate")

	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	# read data
	rdd = sc.pickleFile("/user/mjoys/user_vector1/*/*")
	df = rdd.toDF(["imei", "pkg", "cnt_freq"])
	
	summary_table = process_concated_table(df)

	print("Finished.")	
	print("Writing...")
	summary_table.write.parquet("/user/mjoys/user_vector3")
	print("Get!")
	end_time = time.time()
	delta = end_time - start_time
	print("total cost %s" % delta)
