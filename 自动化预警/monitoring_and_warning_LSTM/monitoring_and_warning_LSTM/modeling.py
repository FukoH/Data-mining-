#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'alison'

"""
导入必要的机器学习和深度学习的库
"""
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


"""
载入指定路径的数据，usecols=[1] 读取第二列
"""
from set_rawdata_path import data_path
dataframe = pandas.read_csv(data_path, usecols=[1], engine='python')
#取dataframe中的数值
dataset = dataframe.values
#将数值类型转换成浮点型
dataset = dataset.astype('float32')


"""
定义一个array的值转换成矩阵的函数
"""
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

"""
分隔训练集和测试集合
"""

# split into train and test sets
size_a=0.67
from set_train_size import size_a
#from set_train_size import size_b

train_size = int(len(dataset) *size_a)
test_size = len(dataset) - train_size
#test_size = int(len(dataset) *(1-size_b))
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#train, test = dataset[0:train_size,:], dataset[test_size:,:]

"""
将数据转换成模型需要的形状，X=t and Y=t+1
"""
look_back = 1
from set_super_parameter import look_back
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
"""
将数据转换成模型需要的形状，[样本samples,时间步 time steps, 特征features]
"""
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

"""
搭建LSTM神经网络
"""
sample =4
nb_epoch=100
optimizer='adam'
from set_super_parameter import sample
from set_super_parameter import nb_epoch
from set_super_parameter import optimizer
model = Sequential()
model.add(LSTM(sample, input_dim=look_back))
#model.add(layers.Dropout(0.01))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=1, verbose=2)

"""
预测数据，对训练集和测试集上的数据进行预测
"""
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
"""
把归一化的预测数据，转换成业务数据的范围和格式
"""
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
"""
模型评估，计算均方根误差RMSE( root mean squared error)
"""
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

"""
把训练集合上的预测结果做成图
"""
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
"""
把测试集合上的预测结果做成图
"""
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
"""
把原始数据作为基线做成图
"""
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

"""
保存训练好的模型
"""
print( u"保存LSTM模型")
from sklearn.externals import joblib
from set_model_result_path import LSTM_model_path
joblib.dump(model,LSTM_model_path)
lasso_model = joblib.load(LSTM_model_path)


