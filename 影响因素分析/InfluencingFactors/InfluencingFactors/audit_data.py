#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'alison'
#导入数据路径
#导入指标名
from data_path import df_path
from data_field_names import df_names

#导入python数据分析库
import pandas as pd
import numpy as np

#读取数据
df = pd.read_table(df_path,sep='|',header = None,names =df_names )
#打印数据行列数
print "df.shape",df.shape

#稽核数据c：统计个数、平均值、标准差、最小值、最大值、空值、不同值个数、10%、25%、50%、75%、90%处的值
#count、mean、std、min、max、null_count、nunique、pct_10、pct_25、pct_50、pct_75、pct_90
result= (df.describe().T).drop(['25%','50%','75%'],axis=1).assign(
           nunique = df.apply(lambda x: x.nunique()),
           pct_10=df.apply(lambda x: x.dropna().quantile(.1)),
           pct_25=df.apply(lambda x: x.dropna().quantile(.25)),
           pct_50=df.apply(lambda x: x.dropna().quantile(.5)),
           pct_75=df.apply(lambda x: x.dropna().quantile(.75)),
           pct_90=df.apply(lambda x: x.dropna().quantile(.9)),
           null_count = df.isnull().sum())
#导入稽核数据结果存放路径
from audit_result_path import audit_result_path

#将稽核结果存入指定文件
result.to_csv(audit_result_path)