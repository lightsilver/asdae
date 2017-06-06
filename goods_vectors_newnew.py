# coding: utf-8

"""
对商品静态数据进行处理，主要针对大批数据量进行park分布式操作
"""

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
conf = SparkConf().setAppName("NewGoodsItemsVectors")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# ---------- 读取数据及预处理 ----------
rdd = sqlContext.read.parquet("/sourcedata/linzhen/20170511_01/")
df = rdd.toPandas()
drop_field = ['MID', 'GOODSIMGURL', 'GOODSURL', 'TBK_SHORT_LINK', 'TBK_LINK', 'TKL', 'TICKET_TIME_START', 'TICKET_TIME_END', 'TICKET_LINK', 'TICKET_TKL', 'TICKET_SHORT_LINK', 'GOODS_LOGO', 'CAMPAIGN_LOGO', 'SLOGAN', 'brand', 'TIME_CREATE', 'TIME_MODIFIED']
df.drop(drop_field, axis=1, inplace=True)

# ---------- 处理文本特征 ----------
import gensim
import jieba
import numpy as np

# 对输入文本列进行向量化处理

# 返回句子向量
def get_vectors(sentence, vectors, vector_size=100):
    """
    对一句话进行词向量的计算
    :param sentence: 分词后的句子
    :param vectors: 训练好的语料词向量
    :param vector_size: 词向量的维度
    :return: 句子的词向量
    """
    sentence_vectors = np.zeros(vector_size)
    for word in sentence:
        try:
            sentence_vectors += vectors[word]
        except:
            continue
    return sentence_vectors

def convert2vectors(data, col, vector_size=100):
    """
    对文本数据进行词向量训练
    :param data: 待处理的数据
    :param col: 待处理的列
    :param vector_size: 向量大小
    :return: 
    """
    # 分词
    if col not in data.columns:
        return "Error: Column is not in exists data!"

    sentences = map(lambda s: jieba.lcut(s, cut_all=False), data[col])

    # 训练语料
    model = gensim.models.Word2Vec(sentences, workers=20)
    words_vectors = model.wv

    # 对句子进行向量转化
    sentences_vectors = []
    # 迭代每句话进行词向量转换
    for sentence in sentences:
        sentences_vectors.append(get_vectors(sentence, words_vectors, vector_size))

    return sentences_vectors

# 对GOODSNAME, SHOPNAME, ALIWW, KEYWORD, ATTRS列进行词向量抽取
# 由于ATTRS是json格式，我们要把其中的value取出进行处理
import json

def extract_values(d):
    """
    对json格式的数据进行values提取
    :param d: json数据
    :return: 
    """
    try:
        values = json.loads(d.encode('utf-8'))
        # 转码
        return " ".join(map(lambda x: x.encode('utf-8'), values))
    except:
        return ''

# 对ATTRS进行values提取
df['ATTRS_TEXT'] = map(extract_values, df['ATTRS'])
df.drop('ATTRS', axis=1, inplace=True)

# 对所有文本进行词向量提取
text_features = ['GOODSNAME', 'SHOPNAME', 'ALIWW', 'KEYWORD', 'ATTRS_TEXT']

for each in text_features:
    # 定义新列的名字
    nel_col = each + '_VECTORS'
    df[nel_col] = convert2vectors(df, each)
    df.drop(each, axis=1, inplace=True)

# --------- 处理数值特征 ----------
def change_datatype(x):
    """
    对数值型特征进行数据类型转换
    :param x: 单个数据
    :return: 转换类型后的数据
    """
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

# ---------- 增加新的属性列 ----------
# (1) 优惠券使用率
df['TICKET_USED'] = (df['TICKET_STOCK'] - df['TICKET_REMAIN_STOCK']) / df['TICKET_STOCK']
df.drop(['TICKET_STOCK', 'TICKET_REMAIN_STOCK'], axis=1, inplace=True)

# (2) 优惠力度
def get_ticket_ratio(s):
    """
    对优惠价格列进行特征生成，提取里面的数值计算优惠力度
    :param s: 原始输入字符串
    :return: 优惠力度
    """
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

# ---------- 将向量转换成多个列 ----------

df['GOODSNAME_STR'] = map(lambda x: ",".join(map(str, x)), df['GOODSNAME_VECTORS'])
df.drop('GOODSNAME_VECTORS', axis=1, inplace=True)
df['SHOPNAME_STR'] = map(lambda x: ",".join(map(str, x)), df['SHOPNAME_VECTORS'])
df.drop('SHOPNAME_VECTORS', axis=1, inplace=True)
df['ALIWW_STR'] = map(lambda x: ",".join(map(str, x)), df['ALIWW_VECTORS'])
df.drop('ALIWW_VECTORS', axis=1, inplace=True)
df['KEYWORD_STR'] = map(lambda x: ",".join(map(str, x)), df['KEYWORD_VECTORS'])
df.drop('KEYWORD_VECTORS', axis=1, inplace=True)
df['ATTRS_STR'] = map(lambda x: ",".join(map(str, x)), df['ATTRS_TEXT_VECTORS'])
df.drop('ATTRS_TEXT_VECTORS', axis=1, inplace=True)


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

# ---------- 将类别转换为one-hot ----------
df = pd.concat([df, pd.get_dummies(df['CATID'], prefix='CATID')], axis=1)
df.drop('CATID', axis=1, inplace=True)

# ---------- 扔掉前三列ID ----------
df.drop(['ID', 'ADVID', 'GOODSID'], axis=1, inplace=True)
print df.head()

# 写出文件
#with open('goods_vectors_newnew.csv', 'a') as f:
#        for row in df.values:
#                f.write(",".join(map(str, row)) + '\n')
#sqldf = sqlContext.createDataFrame(df)
#sqldf.save(path='/user/mjoys/goods_vectors_new', mode='overwrite')
print df.info()

print("Writing...")
# 将数据转化成libsvm
from sklearn.datasets import dump_svmlight_file

dump_svmlight_file(df.values, df.index.values, 'goods_vectors.libsvm')

print("Done")
