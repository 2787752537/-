# -*- coding: utf-8 -*

from os import listdir
import numpy as np
import warnings
from jieba import cut  # 切词
from sklearn.feature_extraction.text import TfidfVectorizer  # 计算单词 TF-IDF 向量的值。
from sklearn.naive_bayes import BernoulliNB  # 伯努利朴素叶贝斯模型
import pandas as pd

warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text = open(file_path, 'r', encoding='UTF-8-sig').read()
    textcut = cut(text)
    # 过滤长度为1的词
    textcut = filter(lambda word: len(word) > 1, textcut)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def getWordsFromFile(file_dir):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :return: 分词后的文档列表
    """
    file_list = listdir(file_dir)
    words_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
    return words_list


#读取训练集
mescon_all = pd.read_csv('./result.csv', header=None, encoding='gbk')
# 导入并解析文本文件
classList = []
fullText = []
for i,line in enumerate(mescon_all[1]):
    fullText.append(mescon_all[1][i])
    classList.append(mescon_all[0][i])
train_words_list = fullText
train_labels = np.array(classList)

# 读入停止词
stop_words = open('./中文邮件/stop/stopword.txt', 'r', encoding='UTF-8-sig').read()
stop_words = stop_words.split('\n')  # 根据分隔符分隔
# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_words_list)

# 贝叶斯分类器
clf = BernoulliNB(fit_prior=False, alpha=0.01).fit(
    train_features, train_labels)

text = pd.read_csv('./result_text.csv', encoding='gbk')
outfile = open('.\submission1.csv','w', encoding='utf-8')
outfile.write("id" + ',' + "label" + '\n')
test_words_list = []
for i in  range(len(text)):
    test_words_list.append(text.comment[i])
test_features = tf.transform(test_words_list)
predicted_labels = clf.predict(test_features)
for docIndex in range(len(text)):
    outfile.write((text.id[docIndex]+','+str(predicted_labels[docIndex])+'\n'))
outfile.flush()
outfile.close()

# 计算准确率
print('训练集精度：', clf.score(train_features, train_labels))