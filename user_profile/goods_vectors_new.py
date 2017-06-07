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

# read file
try:
	sc.stop()
except:
        pass
conf = SparkConf().setAppName("NewGoodsItemsVectors")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

rdd = sqlContext.read.parquet("/sourcedata/linzhen/20170511_01/")

# preprocess
df = rdd.toPandas()

# drop some useless columns
drop_field = ['MID', 'GOODSIMGURL', 'GOODSURL', 'TBK_SHORT_LINK', 'TBK_LINK', 'TKL', 'TICKET_TIME_START', 'TICKET_TIME_END', 'TICKET_LINK', 'TICKET_TKL', 'TICKET_SHORT_LINK', 'GOODS_LOGO', 'CAMPAIGN_LOGO', 'SLOGAN', 'brand', 'TIME_CREATE', 'TIME_MODIFIED']
df.drop(drop_field, axis=1, inplace=True)

# 处理文本特征
import gensim
import jieba
import numpy as np

# 声明词向量
words_vectors = dict()

# 返回句子向量
def get_goodsname_vectors(sentence, vector_size=100):
	sentence_vectors = np.zeros(vector_size)
	for word in sentence:
		try:
			sentence_vectors += words_vectors[word]
		except:
			continue
	return sentence_vectors

# (1) 向量化goodsname
df['GOODSNAME_CORPUS'] = map(lambda x: jieba.lcut(x, cut_all=False)  ,df['GOODSNAME'])
sentences = df['GOODSNAME_CORPUS'].values.tolist()
model = gensim.models.Word2Vec(sentences,workers=20)
words_vectors = model.wv
df['GOODSNAME_VECTORS'] = map(get_goodsname_vectors, df['GOODSNAME_CORPUS'])
df.drop(['GOODSNAME', 'GOODSNAME_CORPUS'], axis=1, inplace=True)
del sentences
del model
del words_vectors

# (2) 向量化shopname
df['SHOPNAME_CORPUS'] = map(lambda x: jieba.lcut(x, cut_all=False)  ,df['SHOPNAME'])
sentences = df['SHOPNAME_CORPUS'].values.tolist()
model = gensim.models.Word2Vec(sentences,workers=20)
words_vectors = model.wv
df['SHOPNAME_VECTORS'] = map(get_goodsname_vectors, df['SHOPNAME_CORPUS'])
df.drop(['SHOPNAME', 'SHOPNAME_CORPUS'], axis=1, inplace=True)
del sentences
del model
del words_vectors

# (3) 向量化aliww
df['ALIWW_CORPUS'] = map(lambda x: jieba.lcut(x, cut_all=False)  ,df['ALIWW'])
sentences = df['ALIWW_CORPUS'].values.tolist()
model = gensim.models.Word2Vec(sentences,workers=20)
words_vectors = model.wv
df['ALIWW_VECTORS'] = map(get_goodsname_vectors, df['ALIWW_CORPUS'])
df.drop(['ALIWW', 'ALIWW_CORPUS'], axis=1, inplace=True)
del sentences
del model
del words_vectors

# (4) 向量化keyword
df['KEYWORD_CORPUS'] = map(lambda x: jieba.lcut(x, cut_all=False)  ,df['KEYWORD'])
sentences = df['KEYWORD_CORPUS'].values.tolist()
model = gensim.models.Word2Vec(sentences,workers=20)
words_vectors = model.wv
df['KEYWORD_VECTORS'] = map(get_goodsname_vectors, df['KEYWORD_CORPUS'])
df.drop(['KEYWORD', 'KEYWORD_CORPUS'], axis=1, inplace=True)
del sentences
del model
del words_vectors

# (5) 向量化ATTRS
# 取出所有value值
import json
def extract_values(d):
	try:
		values = json.loads(d.encode('utf-8'))
		# 转码
		return " ".join(map(lambda x: x.encode('utf-8'), values))
	except:
		return ''
df['ATTRS_TEXT'] = map(extract_values, df['ATTRS'])
df['ATTRS_CORPUS'] = map(lambda x: jieba.lcut(x, cut_all=False)  ,df['ATTRS_TEXT'])
sentences = df['ATTRS_CORPUS'].values.tolist()
model = gensim.models.Word2Vec(sentences,workers=20)
words_vectors = model.wv
df['ATTRS_VECTORS'] = map(get_goodsname_vectors, df['ATTRS_CORPUS'])
df.drop(['ATTRS', 'ATTRS_CORPUS', 'ATTRS_TEXT'], axis=1, inplace=True)
del sentences
del model
del words_vectors

# 处理数值特征
def change_datatype(x):
	try:
		return float(x.encode('utf-8'))
	except:
		return np.NaN

continuous_features = ['GOODSPRICE', 'GOODS_MONTH_SALES', 'INCOME_RATE', 'COMMISION', 'TICKET_STOCK', 'TICKET_REMAIN_STOCK', 'STATUS']

for each in continuous_features:
	df[each] = map(change_datatype, df[each])

# 填充缺失值
df['GOODSPRICE'].fillna(np.mean(df['GOODSPRICE']), inplace=True)
df['GOODS_MONTH_SALES'].fillna(np.mean(df['GOODS_MONTH_SALES']), inplace=True)
df['INCOME_RATE'].fillna(np.mean(df['INCOME_RATE']), inplace=True)
df['COMMISION'].fillna(np.mean(df['COMMISION']), inplace=True)
df['TICKET_STOCK'].fillna(-1, inplace=True)
df['TICKET_REMAIN_STOCK'].fillna(-1, inplace=True)
df['STATUS'].fillna(-1, inplace=True)

# 增加新的属性列
# (1) 优惠券使用率
df['TICKET_USED'] = (df['TICKET_STOCK'] - df['TICKET_REMAIN_STOCK']) / df['TICKET_STOCK']
df.drop(['TICKET_STOCK', 'TICKET_REMAIN_STOCK'], axis=1, inplace=True)

# (2) 优惠力度
def get_ticket_ratio(s):
	prices = re.findall(ur"\d+", s)
	prices = map(float, prices)
	length = len(prices)
	if length == 1:
		return 1.0
	elif length == 2:
		return prices[1] / prices[0]
	else:
		return 0.0

# 对TICKET_PRICE进行空值填充，防止map失败
df['TICKET_PRICE'].fillna(u"", inplace=True)
df['TICKET_RATIO'] = map(get_ticket_ratio, df['TICKET_PRICE'])
df.drop('TICKET_PRICE', axis=1, inplace=True)
# (3) 月销售额
df['MONTH_TOTAL_SALES'] = df['GOODSPRICE'] * df['GOODS_MONTH_SALES']

# 对连续值进行归一化处理
from sklearn.preprocessing import scale
scaleing_features = ['GOODSPRICE', 'GOODS_MONTH_SALES', 'INCOME_RATE', 'MONTH_TOTAL_SALES', 'COMMISION']
for each in scaleing_features:
	df[each] = scale(df[each])

# 将向量转换成多个列

df['GOODSNAME_STR'] = map(lambda x: ",".join(map(str, x)), df['GOODSNAME_VECTORS'])
df.drop('GOODSNAME_VECTORS', axis=1, inplace=True)
df['SHOPNAME_STR'] = map(lambda x: ",".join(map(str, x)), df['SHOPNAME_VECTORS'])
df.drop('SHOPNAME_VECTORS', axis=1, inplace=True)
df['ALIWW_STR'] = map(lambda x: ",".join(map(str, x)), df['ALIWW_VECTORS'])
df.drop('ALIWW_VECTORS', axis=1, inplace=True)
df['KEYWORD_STR'] = map(lambda x: ",".join(map(str, x)), df['KEYWORD_VECTORS'])
df.drop('KEYWORD_VECTORS', axis=1, inplace=True)
df['ATTRS_STR'] = map(lambda x: ",".join(map(str, x)), df['ATTRS_VECTORS'])
df.drop('ATTRS_VECTORS', axis=1, inplace=True)


array = map(lambda x: map(float, x.split(",")), df['GOODSNAME_STR'].values)
df = pd.concat([df, pd.DataFrame(array, columns=map(lambda x: 'GOODSNAME' + str(x), range(100)))], axis=1)
df.drop("GOODSNAME_STR", axis=1, inplace=True)
del array

array = map(lambda x: map(float, x.split(",")), df['SHOPNAME_STR'].values)
df = pd.concat([df, pd.DataFrame(array, columns=map(lambda x: 'SHOPNAME' + str(x), range(100)))], axis=1)
df.drop("SHOPNAME_STR", axis=1, inplace=True)
del array

array = map(lambda x: map(float, x.split(",")), df['ALIWW_STR'].values)
df = pd.concat([df, pd.DataFrame(array, columns=map(lambda x: 'ALIWW' + str(x), range(100)))], axis=1)
df.drop("ALIWW_STR", axis=1, inplace=True)
del array

array = map(lambda x: map(float, x.split(",")), df['KEYWORD_STR'].values)
df = pd.concat([df, pd.DataFrame(array, columns=map(lambda x: 'KEYWORD' + str(x), range(100)))], axis=1)
df.drop("KEYWORD_STR", axis=1, inplace=True)
del array

array = map(lambda x: map(float, x.split(",")), df['ATTRS_STR'].values)
df = pd.concat([df, pd.DataFrame(array, columns=map(lambda x: 'ATTRS' + str(x), range(100)))], axis=1)
df.drop("ATTRS_STR", axis=1, inplace=True)
del array

# 将类别转换为one-hot
df = pd.concat([df, pd.get_dummies(df['CATID'], prefix='CATID')], axis=1)
df.drop('CATID', axis=1, inplace=True)
print df.head()

# 写出文件
with open('goods_vectors_new.csv', 'a') as f:
	for row in df.values:
		f.write(",".join(map(str, row)) + '\n')
#sqldf = sqlContext.createDataFrame(df)
#sqldf.save(path='/user/mjoys/goods_vectors_new', mode='overwrite')

print("Done")
