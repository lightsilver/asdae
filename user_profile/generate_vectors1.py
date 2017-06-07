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

# 1 list all parquet files
def get_parquet_list():
	"""
	para@path: parquet路径
	return@: parquet文件列表
	"""
	cmd = "hadoop dfs -ls /sourcedata/linzhen/20170419_02/"
	result = os.popen(cmd).readlines()[2:]	
	pattern = re.compile(r"(\/sourcedata.*?parquet)")


	parquet_files = map(lambda x: re.search(pattern, x.strip("\n")).group(1), result) 
	
	return parquet_files

# Filter the sample
def process_user_history(df):
        """
        para@df: DataFrame类型数据
        return@: 返回json格式
        """
        df = df.drop("time")
	df = df.toPandas()
	pattern = re.compile(r"^\d{15}$")
	df = df[df["imei"].map(lambda x: bool(re.match(pattern, x)))]
	sqlCtx = SQLContext(sc)
	df = sqlCtx.createDataFrame(df)
        # 3 do the operation
        df_count = df.groupBy("imei", "packagename").count()
	df_count = df_count.selectExpr("imei as imei","packagename as pkg","count as cnt_freq")
        return df_count
        print("Output done.")

def process_concated_table(table):
	"""
	对聚合后的表进行二次统计
	"""
	df_group = table.groupBy("imei", "pkg").agg({"cnt_freq": "sum"})
	df_group = df_group.selectExpr("imei as imei", "pkg as pkg", "`sum(cnt_freq)` as cnt_freq")
	return df_group

# 2 process each parquet

if __name__ == "__main__":
	from_idx, to_idx = map(int,  sys.argv[1:])
	#from_idx = 100
	#to_idx = 150
		
	# get all file name
	print("read list...")
	parquet_list = get_parquet_list()
	print("Done.")
	#sparkConf.set("spark.driver.allowMultipleContexts","true")
	
	start_time = time.time()
	try:
		sc.stop()
	except:
		pass
	conf = SparkConf().setAppName("UserLabelGenerate")
	#set("spark.driver.allowMultipleContexts", "true")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	# define the summary table
	cnt = 1
	for pf in parquet_list[from_idx:to_idx]:
		pt = sqlContext.read.parquet(pf)
		if cnt == 1:
			summary_table = process_user_history(pt)
		else:
			summary_table = summary_table.unionAll(process_user_history(pt))
			summary_table = process_concated_table(summary_table)
		
		print("%s File Has Done." % cnt)
		cnt += 1
		#print(summary_table.count())	

	print("End.")	
	print("Writing...")
	summary_table.rdd.map(tuple).saveAsPickleFile("/user/mjoys/user_vector/total_summary" + str(from_idx/50), batchSize=1)
	print("Get!")
	end_time = time.time()
	delta = end_time - start_time
	print("total cost %s" % delta)
	#sc.stop()	
