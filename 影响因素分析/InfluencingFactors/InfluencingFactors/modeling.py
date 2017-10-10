#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'alison'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#nohup python modeling.py >    /data/quota/zbjk/modeling_result.log 2>&1 &
#导入python数据分析库
import pandas as pd
import numpy as np
import sklearn

print(u"导入数据")
#导入建模数据
from  data_preprocessing import df_new
#导入目标指标和训练集删除指标：
from set_classification_variable import drop_variable,target_variable
#导入分类指标名字
from set_classification_variable import classification_variable
#将分类变量转换成字符串
for i in classification_variable:
    df_new[i] = df_new[i].astype(str)

#删除以下指标，作为训练数据
df_new_drop =df_new.drop(drop_variable,axis = 1)

print u"df_new_drop是否有空值(无-False)："
print np.any(np.isnan(df_new_drop))
print u"df_new_drop是否数据有限大（有限大-True，无穷大-False，）："
print np.all(np.isfinite(df_new_drop))

print u"标准化y"
data_frame = pd.DataFrame({'A': [1.1, 2, 3.1],
                   'B': [1, 2, 3],
                   'C': ["A1","A2", "A3"]})
if type(df_new[target_variable][0]) == type(data_frame['A'][0]) or type(df_new[target_variable][0]) == type(data_frame['B'][0]):
    y_data_mean =df_new[target_variable].mean()
    y_data_std =df_new[target_variable].std()
    y_train_1 = (df_new[target_variable]-y_data_mean)/y_data_std
    y_train = y_train_1.values
else:
    y_train  =df_new.loc[:,target_variable].values
print u"y_train是否有空值(无-False)："
print np.any(np.isnan(y_train))
print u"y_train是否数据有限大（有限大-True，无穷大-False，）："
print np.all(np.isfinite(y_train))
#生成索引对
#
def unique_pairs():
    for i in classification_variable:
        for j in df_new_drop.columns:
            yield i, j

for i, j in unique_pairs():
    if  i == j:
        df_dummy = pd.get_dummies(df_new_drop)
        break
    else:
        df_dummy = df_new_drop
        break

#49列
#数据备份一下
df_dummy_2 = df_dummy
print("df_dummy_2",df_dummy_2.shape)
print u"标准化X"
#删除标准差为0的指标
#筛选数值型指标列
numeric_cols = df_dummy_2.columns[df_dummy_2.dtypes != 'object']
numeric_col_std = df_dummy_2.loc[:, numeric_cols].std()

std_0_list =[]
for i in numeric_col_std.index:
    if numeric_col_std[i] ==0:
        std_0_list.append(i)

print std_0_list
print "df_dummy_2",df_dummy_2.shape
#更新数据集
df_dummy_2 =df_dummy_2.drop(std_0_list,axis = 1)
print "df_dummy_2",df_dummy_2.shape

#更新数值指标列表
numeric_cols = df_dummy_2.columns[df_dummy_2.dtypes != 'object']

#标准化处理
numeric_col_means = df_dummy_2.loc[:, numeric_cols].mean()
numeric_col_std = df_dummy_2.loc[:, numeric_cols].std()
df_dummy_2.loc[:, numeric_cols] = (df_dummy_2.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

X_train = df_dummy_2.values
#a = df_new['lastmonth_real_fee'][1]
print u"标准化完成"
print("X_train",X_train.shape)
print u"是否有空值(无-False)："
print np.any(np.isnan(df_dummy_2))
print u"是否数据有限大（有限大-True，无穷大-False，）："
print np.all(np.isfinite(df_dummy_2))
#dtypes
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
print u"开始建模"
# 总得切分一下数据咯（训练集和测试集）

cv = cross_validation.ShuffleSplit(len(X_train), n_iter=3, test_size=0.2,
    random_state=0)

lasso = linear_model.LassoCV()
lasso.fit(X_train, y_train)
print(u"lasso训练：自动生成最优超参数")
print(lasso.alpha_)
#0.000994206438274
best_alpha = lasso.alpha_

print(u"用最优超参数建模lasso")
for train, test in cv:
    lasso_svc = linear_model.Lasso(alpha = best_alpha).fit(X_train[train], y_train[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        lasso_svc.score(X_train[train], y_train[train]), lasso_svc.score(X_train[test], y_train[test])))
#####输出模型

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        #names = ["X%s" % x for x in range(len(coefs))]
        names = [x for x in df_dummy_2.columns]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return  " + ".join("%s * %s" % (round(coef, 3), name)  for coef, name in lst)
print "Linear model lasso:", pretty_print_linear(lasso_svc.coef_)

from set_model_result_path import lasso_model_path
print(u"保存lasso模型")
from sklearn.externals import joblib
joblib.dump(lasso_svc,lasso_model_path)
lasso_model = joblib.load(lasso_model_path)

###############################################################################
print(u"GradientBoostingRegressor训练")
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


X_train_1 = X_train.astype(np.float32)
offset = int(X_train_1.shape[0] * 0.9)
X_train_data, y_train_data = X_train_1[:offset], y_train[:offset]
X_test_data, y_test_data = X_train_1[offset:], y_train[offset:]

# Fit regression model
params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf_gbr = ensemble.GradientBoostingRegressor(**params)

clf_gbr.fit(X_train_data, y_train_data)
mse = mean_squared_error(y_test_data, clf_gbr.predict(X_test_data))
print("MSE: %.4f" % mse)

print u"计算测试集误差"
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf_gbr.staged_predict(X_test_data)):
    test_score[i] = clf_gbr.loss_(y_test_data, y_pred)


feature_importance = clf_gbr.feature_importances_
#feature_importance= feature_importance*100
feature_importance = 100.0 * (feature_importance-feature_importance.min() )/ (feature_importance.max()-feature_importance.min())
sorted_idx = np.argsort(feature_importance)

print u"输出指标及其重要性"

for i in range(len(df_dummy_2.columns)):
    print df_dummy_2.columns[sorted_idx][i],",",feature_importance[sorted_idx][i]

print u"保存GradientBoostingRegressor模型"
from sklearn.externals import joblib
from set_model_result_path import gbr_model_path
joblib.dump(clf_gbr,gbr_model_path)
lasso_model = joblib.load(gbr_model_path)
