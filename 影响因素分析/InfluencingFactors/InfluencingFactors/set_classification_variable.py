#!/usr/bin/python
# -*- coding: utf-8 -*

"""
将建模数据中的所有分类变量放入下面中括号里：
举例：
是否出账	bill_flag
是否月活跃	active_flag
是否过网	ctc_amt
"""
classification_variable = ['bill_flag','active_flag','active_flag']

"""
训练集删除指标：
预测指标中，需要删掉的指标：一般需要删掉user_id，以及目标指标：比如：研究出账用户数，则训练集需要删除用户是否出账：bill_flag
"""

drop_variable = [
    "user_id_a",
"bill_flag",
"lastmonth_real_fee",
"active_flag",
"fee",
"cur_mon_owe_fee",
"ctc_amt"
]

"""
目标指标：（只能有一个）

比如：研究出账用户数，目标指标为用户是否出账：bill_flag
再比如：研究本月实收上期费用(元)	lastmonth_real_fee 则该指标为目标指标
"""

target_variable = "lastmonth_real_fee"

