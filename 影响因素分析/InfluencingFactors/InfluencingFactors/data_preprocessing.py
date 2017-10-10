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
import sklearn

#读取数据
df = pd.read_table(df_path,sep='|',header = None,names =df_names )
#打印数据行列数
print u"读取数据成功！数据行列数df.shape为",df.shape
#备份并将数据存储到新变量
df_new =df

from set_audit_dictionary import all_null,dict_fill
if all_null == 1:
    df_new= df.fillna(0)
else:
    for key in dict_fill:
        df_new.loc[:,dict_fill[key]['columns']] = df_new.loc[:,dict_fill[key]['columns']].fillna(dict_fill[key]['values'])

#查看是否填充完全，打印结果应该为0，则说明空值已经全部填充完毕
print(df_new.isnull().sum())

print u"是否有空值(无-False)："
print np.any(np.isnan(df_new))
print u"是否数据有限大（有限大-True，无穷大-False，）："
print np.all(np.isfinite(df_new))

print(u"数据中空值处理完毕")
#稽核填充成0之后的新数据
#导入稽核填充后数据结果存放路径
from audit_result_path import audit_result_path_fill
result_fill= (df_new.describe().T).drop(['25%','50%','75%'],axis=1).assign(
           nunique = df_new.apply(lambda x: x.nunique()),
           pct_10=df_new.apply(lambda x: x.dropna().quantile(.1)),
           pct_25=df_new.apply(lambda x: x.dropna().quantile(.25)),
           pct_50=df_new.apply(lambda x: x.dropna().quantile(.5)),
           pct_75=df_new.apply(lambda x: x.dropna().quantile(.75)),
           pct_90=df_new.apply(lambda x: x.dropna().quantile(.9)),
           null_count = df_new.isnull().sum())
#将稽核结果保存
result_fill.to_csv(audit_result_path_fill)

print  u"空值处理完成后，第二次基础统计分析已经存入到:",audit_result_path_fill