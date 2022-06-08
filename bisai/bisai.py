import random
from numpy import *
import pandas as pd
# 去列表中重复元素，并以列表形式返回
def createVocaList(dataSet):
    vocabSet = set({})
    # 去重复元素，取并集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 统计每一文档（或邮件）在单词表中出现的次数，并以列表形式返回
def setOfWordsToVec(vocabList, inputSet):
    # 创建0向量，其长度为单词量的总数
    returnVec = [0] * len(vocabList)
    # 统计相应的词汇出现的数量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    # 获取训练文档数
    numTrainDocs = len(trainMatrix)
    # 获取每一行词汇的数量
    numWords = len(trainMatrix[0])
    # 侮辱性概率(计算p(Ci))，计算垃圾邮件的比率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 统计非垃圾邮件中各单词在词数列表中出现的总数（向量形式）
    p0Num = ones(numWords)
    # 统计垃圾邮件中各单词在词数列表中出现的总数（向量形式）
    p1Num = ones(numWords)
    # 统计非垃圾邮件总单词的总数（数值形式）
    p0Denom = 2.0
    # 统计垃圾邮件总单词的总数（数值形式）
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 如果是垃圾邮件
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 如果是非垃圾邮件
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算每个单词在垃圾邮件出现的概率（向量形式）
    p1Vect = log(p1Num / p1Denom)
    # 计算每个单词在非垃圾邮件出现的概率（向量形式）
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return '1'
    else:
        return '0'


# test
def spamtest():
    # 导入并解析文本文件
    docList = []
    classList = []
    fullText = []

    mescon_all = pd.read_csv('./result.csv', header=None, encoding='gbk')
    count = 0
    for i,line in enumerate(mescon_all[1]):
        try:
            if len(line) != 0:
                wordList = line.split()
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(mescon_all[0][i])
                count += 1
        except:
            continue
    print(fullText)
    # 去除重复的元素
    # vocabList = docList
    vocabList = createVocaList(docList)
    print(vocabList)
    trainingSet = [x for x in range(int(count))]
    # 测试集，选10篇doc
    testSet = []
    # 选出测试集
    for i in range(int(0.3 * count)):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    # 选出训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWordsToVec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #开始训练参数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 对测试集分类
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWordsToVec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("错误率为：{0}".format(float(errorCount) / len(testSet)))
    print("正确率为：{0}".format(1- float(errorCount) / len(testSet)))

def spamtest1():
    # 导入并解析文本文件
    docList = []
    classList = []
    fullText = []

    mescon_all = pd.read_csv('./result.csv', header=None, encoding='gbk')
    count = 0
    for i,line in enumerate(mescon_all[1]):
        try:
            if len(line) != 0:
                wordList = line.split()
                docList.append(wordList)
                fullText.extend(wordList)
                classList.append(mescon_all[0][i])
                count += 1
        except:
            continue
    # 去除重复的元素
    # vocabList = docList
    vocabList = createVocaList(docList)

    trainMat = []
    trainClasses = []
    # 选出训练集
    for docIndex in range(count):
        trainMat.append(setOfWordsToVec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #开始训练参数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    text = pd.read_csv('./result_text.csv', encoding='gbk')
    outfile = open('.\submission1.csv','w', encoding='utf-8')
    outfile.write("id" + ',' + "label" + '\n')
    # 对测试集分类
    for docIndex in range(len(text)):
        wordVector = setOfWordsToVec(vocabList, text.comment[docIndex])
        outfile.write((text.id[docIndex]+','+classifyNB(array(wordVector), p0V, p1V, pSpam)+'\n'))
    outfile.flush()
    outfile.close()
spamtest1()



