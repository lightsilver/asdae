# coding: utf-8

import os
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import json

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
        df.drop("timestamp", inplace=True, axis=1)
        pattern = re.compile(ur"^\d{15}$")
        df = pd.DataFrame(filter(lambda x: re.match(pattern, x[1]), df.values), columns=['pkg','imei', 'h'])

        # 3 do the operation
        df['count'] = 1
        user_vectors = pd.pivot_table(df, values="count", index=["imei"], columns=["pkg"], aggfunc=np.sum)
        user_vectors.fillna(0, inplace=True)
	print("-" * 30)
	#print(user_vectors)
	
        # 4 dumps the dataframe to dict
        print("Starting transfer...")
        user_label_dict = {}
        df_size = user_vectors.shape
        for i in xrange(df_size[0]):
                idx = user_vectors.ix[i].name
                user_label_dict.setdefault(idx, {})
		# user_label_dict.setdefault('"%s"'%idx, {})
                for col in user_vectors.ix[i].index.values:
                        if user_vectors.ix[i][col] != 0:
                                user_label_dict[idx][col] = user_vectors.ix[i][col]
				

        for k, v in user_label_dict.iteritems():
                print(k)
                for app, freq in v.iteritems():
                        print("\t%s, %s" % (app, freq))

        
	os.system('echo "%s"| hadoop dfs -put - /user/mjoys/test/test666' % (json.dumps(user_label_dict)))
        print("Output done.")

# 2 process each parquet
if __name__ == "__main__":
	# get all file name
	print("read list...")
	parquet_list = get_parquet_list()
	for i in parquet_list:
        	print i
	print("Done.")

	try:
		sc.stop()
	except:
		pass
	conf = SparkConf().setAppName("UserLabelGenerate")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	# define the summary table
	summary_table = pd.DataFrame()
	for pf in parquet_list[-2:]:
		pt = sqlContext.read.parquet(pf)
		pfc = pt.collect()
		# how to process
		df = pd.DataFrame(pfc, columns=["pkg", "imei", "h", "timestamp"])
		summary_table = pd.concat([summary_table, df])
	
	process_user_history(df)

# show the summary table
#print(summary_table)
#print(summary_table.shape)
