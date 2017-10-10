#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'alison'

"""
设置超参数：
如果模型训练结果loss较大（loss为0-1之间的数，越小越好），或者均方根误差RMSE较大（越小越好），则需要调整超参数
"""

look_back = 1
sample =4
nb_epoch=100
optimizer='adam'

"""
目前结果来看：1）look_back = 1结果最优，look_back可以设置成1,2,3等，不建议太大
             2）sample =4    结果最优，sample可以设置成3,4,5,6,7,8等，不建议太大或太小
             3）nb_epoch=100 结果最优，nb_epoch可以迭代成20,50,100,150,200等，太小误差大，太大消耗计算资源
             4）optimizer='adam'，结果最优，也可以尝试Adagrad，Adadelta，Adam，Adamax，Nadam等
"""