#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017/12/30 19:13
kNN: k-近邻算法

Input:  inX: 用于判断类别的矩阵 (1xN)
        dataSet: 已知类别的数据集向量 (NxM)
        labels: 数据集的标签 (1xN)
        k: 用于比较的近邻的数量 (应该是奇数)

Output: 最可能的类标签


@author: Lucifer
@site: plantree.me
@email: wpy1174555847@outlook.com
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# kNN
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # np.tile()将inX重复(dataSetSize, 1)次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 排序后的索引
    sortedDistanceIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        # 寻找出现次数最的标签
        voteILabel = labels[sortedDistanceIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    # 字典的排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

'''
group, labels = createDataSet()
print(classify0([0, 0], group, labels, 3))
'''

'''
改进约会网站的配对效果
'''
# 读取数据
def file2matrix(file_name):
    fr = open(file_name)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化数据
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: {0}, the real answer is： {1}'
              .format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('the total error rate is: {}'.format(errorCount / numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per years?'))
    iceCream = float(input('liters of ice crean consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probaly like this person: {}'.format(resultList[classifierResult - 1]))


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
'''
# 画图
fig = plt.figure()
cord1 = datingDataMat[np.array(datingLabels) == 1]
cord2 = datingDataMat[np.array(datingLabels) == 2]
cord3 = datingDataMat[np.array(datingLabels) == 3]
ax = fig.add_subplot(111)
type1 = ax.scatter(cord1[:, 0], cord1[:, 1], s=20, c='red', label='Dis Not Like')
type2 = ax.scatter(cord2[:, 0], cord2[:, 1], s=30, c='green', label = 'Liked in Small Does')
type3 = ax.scatter(cord3[:, 0], cord3[:, 1], s=40, c='blue', label='Liked in Large Does')
ax.legend(loc=0)
ax.set_xlabel('Frequent Flyier Miles Earned Per Year')
ax.set_ylabel('Percentage of time Spent Playing Video Games')
plt.show()
'''

'''
normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)
'''

'''
datingClassTest()
'''
'''
classifyPerson()
'''

'''
手写识别系统
'''
def img2vec(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#print(img2vec('digits/trainingDigits/0_13.txt'))


def handwritingClassTest():
    # 训练数据
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vec('digits/trainingDigits/{}'.format(fileNameStr))

    # 测试数据
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vec('digits/testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: {}, the real answer is: {}'
              .format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('the totale number of error is: {}'.format(errorCount))
    print('the total error rate is: {}'.format(errorCount / mTest))


handwritingClassTest()