#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018/1/2 9:36

@author: Lucifer
@site: plantree.me
@email: wpy1174555847@outlook.com
"""

from math import log
import operator
import pickle
from treePlotter import *

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # dict中get的始用，避免条件判断
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        # 根据熵计算公式
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 测试数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatureVec = featVec[:axis]
            reducedFeatureVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatureVec)
    return  retDataSet


# 选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 原始熵值
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList =  [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算划分后的熵值
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是熵的减少
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 投票表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同就终止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 特征只有一个的时候投票表决
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 剔除分类过的标签
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # list复制
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree


# 根据决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict:
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 读取决策树
def grapTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)



if __name__ == '__main__':
    '''
    myDat, labels = createDataSet()
    print(calcShannonEnt(myDat))
    myDat[0][-1] = 'maybe'
    print(calcShannonEnt(myDat))
    '''

    '''
    myDat,labels = createDataSet()
    print(splitDataSet(myDat, 0, 1))
    '''

    '''
    myDat, labels = createDataSet()
    print(chooseBestFeatureToSplit(myDat))
    '''

    '''
    myDat, labels = createDataSet()
    #myTree = retrieveTree(0)
    #storeTree(myTree, 'classifierStore.txt')
    myTree = grapTree('classifierStore.txt')
    print(classify(myTree, labels, [1, 1]))
    '''

    # 预测隐形眼镜类型
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lenseLabels)
    createPlot(lensesTree)
    print(lensesTree)