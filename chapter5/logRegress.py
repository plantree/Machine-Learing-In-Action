#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018/1/9 13:38

@author: Lucifer
@site: plantree.me
@email: wpy1174555847@outlook.com
"""

from matplotlib import pyplot as plt
import numpy as np
import random


# 加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat


# Sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = dataMatrix.shape
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBeastFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 计算X1和X2的关系
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升
def stocGradAscent(dataMatrix, classLabel, numIter=150):
    dataIndex = np.array(dataMatrix)
    classLabel = np.array(classLabel)
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent(np.array(trainingSet), np.array(trainingLabels), 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = errorCount / numTestVec
    print('the error rate of this test is: {}'.format(errorRate))
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('alter {} iterations the average error rate is: {}'.format(numTests, errorSum / numTests))


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    '''
#     weights = gradAscent(dataArr, labelMat)
    weights = stocGradAscent(np.array(dataArr), np.array(labelMat))
    print(weights)
    #plotBeastFit(weights.getA())   # matrix.getA()->ndarray
    plotBeastFit(weights)
    '''
    multiTest()
    #colicTest()
    '''
    f = open('horseColicTraining.txt')
    for line in f.readlines():
        item = line.strip().split()
        print(len(item))
    '''