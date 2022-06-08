import pandas as pd
import jieba
import re

train = pd.read_csv('./train.csv',encoding='utf8')

outfile = open('./result.csv','w')
ns = 0
ps = 0
for i in range(len(train)):
    train_ALL = train.loc[i][0]
    train_Y,train_X = train_ALL.split('\t',2)
    outstr = ''
    temp = re.sub(u'[^\u4e00-\u9fa5A-Za-z]','',train_X)
    ms_cut = list(jieba.cut(temp,cut_all=False))
    for word in ms_cut:
        if word != ' ':
            outstr += word+' '
    if train_Y == '1':
        ns = ns+1
        if ns <80000:
            outfile.write((train_Y+','+outstr)+'\n')
    if train_Y == '0':
        ps = ps+1
        if ps <80000:
            outfile.write((train_Y+','+outstr)+'\n')
outfile.flush()
outfile.close()
