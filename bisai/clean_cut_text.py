import pandas as pd
import jieba
import re

text = pd.read_csv('./test_new.csv',encoding='utf8')
header = ["id","comment"]
outfile = open('./result_text.csv','w')
outfile.write(("id"+','+"comment")+'\n')
for i in range(len(text)):
    text_id = text.loc[i][0]
    text_X = text.loc[i][1]
    outstr = ''
    temp = re.sub(u'[^\u4e00-\u9fa5A-Za-z]','',text_X)
    ms_cut = list(jieba.cut(temp,cut_all=False))
    for word in ms_cut:
        if word != ' ':
            outstr += word+' '
    outfile.write((text_id+','+outstr)+'\n')
outfile.flush()
outfile.close()
