import argparse
import os
import random
import re
import shutil
import numpy
import platform
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer #词性还原
from numpy import ones, zeros
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix 
import joblib
import pandas as pd

import pyecharts.options as opts
from pyecharts.charts import Line, Pie
from pyecharts.commons.utils import JsCode
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot as driver

if platform.system()=='Linux':
    from nltk import data
    data.path.append(r"/home/cxm-irene/nltk_data")

class Bayesian():
    def __init__(self, args):
        self.model_choose = args.model
        self.train_file_path = args.train_path
        self.test_file_path = args.test_path
        self.test_times = args.test_times
        self.test_file_num = args.test_file_num
        self.catagory = ['spam', 'ham']
        self.list_stopWords=list(set(stopwords.words('english')))
        self.wrong_file_times = {}
        self.rate = {}
        self.wrong_file = {}
        self.wrong_num = {}
        self.lemmatizer = WordNetLemmatizer()
    
        self.all_file = {}
        self.operation_file = {}
        self.train_file = {}
        self.test_file = {}

        
    def bayes(self):
        self.result_save_init()
        self.get_all_file()
        if self.model_choose == 1:
            for i in range(self.test_times):
                self.turn = i+1
                self.wrong_file.setdefault("第 "+str(self.turn)+" 次测试",{})
                print("第 "+str(self.turn)+" 次测试")

                self.file_recover()
                self.create_test_file()
                self.word_handle()
                self.bayes_paul()
        elif self.model_choose == 2:
            for i in range(self.test_times):
                self.turn = i+1
                self.wrong_file.setdefault("第 "+str(self.turn)+" 次测试",{})
                print("第 "+str(self.turn)+" 次测试")

                self.file_recover()
                self.create_test_file()
                self.word_handle()
                self.bayes_polynomial()
        elif self.model_choose == 3:
            for i in range(self.test_times):
                self.turn = i+1
                self.wrong_file.setdefault("第 "+str(self.turn)+" 次测试",{})
                print("第 "+str(self.turn)+" 次测试")

                self.file_recover()
                self.create_test_file()
                self.word_handle()
                self.SVM()
        elif self.model_choose == 4:
            self.wrong_file.setdefault("第 "+str(self.turn)+" 次测试",{})
            self.create_test_file(True)
            self.word_handle(True)
            print("----------伯努利----------")
            self.bayes_paul()
            print("----------多项式----------")
            self.bayes_polynomial()
            print("-----------SVM-----------")
            self.SVM()
            # self.file_recover()
        else:
            for i in range(self.test_times):
                self.turn = i+1
                self.wrong_file.setdefault("第 "+str(self.turn)+" 次测试",{})
                print("第 "+str(self.turn)+" 次测试")

                self.file_recover()
                self.create_test_file()
                self.word_handle()
                print("----------伯努利----------")
                self.bayes_paul()
                print("----------多项式----------")
                self.bayes_polynomial()
                print("-----------SVM-----------")
                self.SVM()
        print("错误文件：",end="")
        print(self.wrong_file)
        print("错误次数：",end="")
        print(self.wrong_file_times)
        print("正确率与错误率：",end="")
        print(self.rate)

    def get_wordnet_pos(self, treebank_tag):
        # 词性标注提取
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def remove_str(self, sentence):
        rep = {'~':'','`':'','!':'','@':'','#':'','$':'','%':'','^':'','&':'','*':'','(':'',')':'','-':'','_':'','=':'','+':'','[':'','`]':'','{':'','}':'',';':'',':':'','\'':'','\"':'',',':'','<':'','>':'','.':'','/':'','?':'','\n':''}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        #print(rep)
        #print(rep.keys())
        pattern = re.compile("|".join(rep.keys()))
        #print(pattern)
        sentence = pattern.sub(lambda m: rep[re.escape(m.group(0))], sentence)
        sentence = sentence.split(' ')
        
        sentence_ = [x.strip().lower() for x in sentence if x.isalpha() and x.strip()!='' and x.lower() not in self.list_stopWords and len(x)>2]
        sentence_result = []
        for word in sentence_:
            pos_tags = nltk.pos_tag(word)
            _, pos = pos_tags[0]
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            sentence_result.append(self.lemmatizer.lemmatize(word, pos=wordnet_pos))
        
        return sentence_result

    # 提取各文本中的每一个单词并存放在字典中，格式为{'类别':{'文件名':[单词（可重复）]}}
    # 最后返回的是以字典形式存放的单词和去除重复的单词库
    def word_cut(self):
        Lexicon = set([])
        train_words = {}
        train_words.setdefault('spam',{})
        train_words.setdefault('ham',{})
        test_words = {}
        test_words.setdefault('spam',{})
        test_words.setdefault('ham',{})
        for operation, o_value in self.operation_file.items():
            for cata in self.catagory:
                for file in o_value[cata]:
                    with open(file) as f:
                        file_word = []
                        for line in f:
                            if len(line)>1:
                                word_content = self.remove_str(line)
                                file_word.extend(word_content)
                                if operation == 'train':
                                    Lexicon = Lexicon | set(word_content)
                        if operation == 'train':
                            train_words[cata].setdefault(file,file_word)
                        elif operation == 'test':
                            test_words[cata].setdefault(file,file_word)
        return train_words, test_words, list(Lexicon)

    def word_handle(self, to_excel=False):
        self.train_words, self.test_words, self.train_Lexicon = self.word_cut()
        if to_excel:
            df = pd.DataFrame(self.train_words)
            df.to_excel('train_words.xlsx')
            df = pd.DataFrame(self.train_Lexicon)
            df.to_excel('train_Lexicon.xlsx')
            df = pd.DataFrame(self.test_words)
            df.to_excel('test_words.xlsx')

    
    def save_result(self, model, wrong, right, ham_wrong, ham_right, wrong_file):
        self.wrong_num[model].append(wrong-ham_wrong)
        spam_right = round((right-ham_right)/len(self.test_words['spam']),3)
        ham_right = round(ham_right/len(self.test_words['ham']),3)
        spam_wrong = round((wrong-ham_wrong)/len(self.test_words['spam']),3)
        ham_wrong = round(ham_wrong/len(self.test_words['ham']),3)
        right = round(right/(len(self.test_words['spam'])+len(self.test_words['ham'])),3)
        wrong = round(wrong/(len(self.test_words['spam'])+len(self.test_words['ham'])),3)
        self.rate[model]["right"].append(right)
        self.rate[model]["wrong"].append(wrong)
        self.rate[model]["ham-right"].append(ham_right)
        self.rate[model]["ham-wrong"].append(ham_wrong)
        self.rate[model]["spam-right"].append(spam_right)
        self.rate[model]["spam-wrong"].append(spam_wrong)
        self.wrong_file["第 "+str(self.turn)+" 次测试"].setdefault(model, wrong_file)
        print("正确率为："+str(right)+" 错误率为："+str(wrong))
        # print("错误文件为：",end=" ")
        # print(wrong_file)

    # 伯努利模型，步骤为：
    # 1、计算先验概率：P(C)=类C下文档总数/整个训练样本的文档总数   ---> P_S、P_H
    # 2、计算个单词的频率（调整因子）：P(tk|C) = (类C下包含单词tk的文件数+1)/(类c的文档总数+2)  ---> 调用word_proportion_paul函数
    # 3、进行测试   ---> 调用test_word_paul函数
    def bayes_paul(self):
        # words, Lexicon = self.word_cut(self.train_file_path)            #words为字典，存有垃圾邮件和正常邮件的所有单词;Lexicon为词库
        P_S = len(self.train_words['spam'])/(len(self.train_words['spam'])+len(self.train_words['ham']))
        P_H = len(self.train_words['ham'])/(len(self.train_words['spam'])+len(self.train_words['ham']))
        Lexicon_proportion = self.word_proportion_paul(self.train_words, self.train_Lexicon)
        # print(Lexicon_proportion)
        df = pd.DataFrame(Lexicon_proportion)
        df.to_excel('paul.xlsx')
        self.test_word_paul(P_S, P_H, Lexicon_proportion)

    # 伯努利模型下各单词出现的频率，设类别为C，单词为tk，则计算公式为：
    # P(tk|C) = (类C下包含单词tk的文件数+1)/(类c的文档总数+2)
    # 结果最后返回字典，形式为{'类别':{'单词名':频率}}，其中频率保留6位小数点
    def word_proportion_paul(self, words, Lexicon):
        Lexicon_proportion = {}
        Lexicon_proportion.setdefault('spam',{})
        Lexicon_proportion.setdefault('ham',{})
        for l in Lexicon:
            for catagory, c_value in words.items():
                c_l_num = 1
                for _, t_words in c_value.items():
                    if l in t_words:
                        c_l_num += 1
                l_proportion = round(c_l_num/(len(words[catagory])+2),6)
                Lexicon_proportion[catagory].setdefault(l, l_proportion)
        # print(Lexicon_proportion)
        return Lexicon_proportion

    # 伯努利模型下的测试，步骤为：
    # 1、测试集单词分割，与训练集一致
    # 2、进行预测   ---> 调用classify_paul函数
    def test_word_paul(self, P_S, P_H, Lexicon_proportion):
        # test_path = 'test/'
        # test_words = self.word_cut(test_path, False)
        right = 0
        ham_right = 0
        wrong = 0
        ham_wrong = 0
        wrong_file = []
        for catagory, c_value in self.test_words.items():
            for txt, t_words in c_value.items():
                # print("测试文件为："+txt,end=" ")
                classfiy = self.classify_paul(t_words, P_S, P_H, Lexicon_proportion)
                if classfiy == catagory:
                    # print("√√√√√√√√√√判断成功√√√√√√√√√√")
                    right += 1
                    if catagory == "ham":
                        ham_right += 1
                else:
                    # print("！！！！！!判断失败! ！！！！！")
                    wrong += 1
                    if catagory == "ham":
                        ham_wrong += 1
                    wrong_file.append(catagory+txt)
                    if platform.system()=='Windows':
                        self.wrong_file_times["paul"][catagory][int(txt.split('.')[0].split('\\')[-1])-1]+=1
                    elif(platform.system()=='Linux'):
                        self.wrong_file_times["paul"][catagory][int(txt.split('.')[0].split('/')[-1])-1]+=1
        
        self.save_result('paul', wrong, right, ham_wrong, ham_right, wrong_file)

    # 伯努利模型下进行预测，步骤为：
    # 1、公式为：P(C|W1,W2...Wn,Wn+1',Wn+2'...W|V|') = P(C)P(W1,W2...Wn,Wn+1',Wn+2'...W|V|'|C) = P(C)P(W1|C)P(W2|C)...P(Wn|C)(1-P(Wn+1|C))...(1-P(W|V| |C))
    #    其中W1到Wn为进行测试的文档中出现的单词，'表示非，|V|为词库的大小
    # 2、对正常邮件和垃圾邮件的概率分别进行计算，最后谁的概率大即为谁
    def classify_paul(self, words_list, P_H, P_S, Lexicon_proportion):
        P_SH_W = {}
        P_SH_W.setdefault('spam',P_S)
        P_SH_W.setdefault('ham',P_H)
        for catagory, c_value in Lexicon_proportion.items():
            for word_name, word_proportion in c_value.items():
                if word_name in words_list:
                    P_SH_W[catagory] = P_SH_W[catagory]*word_proportion
                else:
                    P_SH_W[catagory] = P_SH_W[catagory]*(1-word_proportion)
        # print(P_SH_W)
        if P_SH_W['spam'] > P_SH_W['ham']:
            # print("为垃圾邮件")
            return 'spam'
        else:
            # print("为正常邮件")
            return 'ham'

    # 多项式模型，步骤为：
    # 1、计算先验概率：P(C)=类C下单词总数/整个训练样本的单词总数   ---> P_S、P_H
    # 2、计算个单词的频率（调整因子）：P(tk|C) = (类C下单词tk在各文档中出现过的次数之和+1)/(类c的单词总数+|V|)，其中|V|为词库大小  ---> 调用words_proportion_polynomial函数
    # 3、进行测试   ---> 调用test_word_paul函数
    def bayes_polynomial(self):
        # words, Lexicon = self.word_cut(self.train_file_path)            #self.train_words为字典，存有垃圾邮件和正常邮件的所有单词;Lexicon为词库
        words_statistics = self.words_statistics_polynomial(self.train_words)
        P_S = words_statistics['spam'] / words_statistics['all']
        P_H = words_statistics['ham'] / words_statistics['all']
        # print(P_S)
        # print(P_H)
        Lexicon_proportion = self.words_proportion_polynomial(self.train_words, self.train_Lexicon, words_statistics)
        df = pd.DataFrame(Lexicon_proportion)
        df.to_excel('poly.xlsx')
        self.test_word_polynomial(P_S, P_H, Lexicon_proportion, self.train_Lexicon)

    # 多项式模型下各单词出现的频率，设类别为C，单词为tk，则计算公式为：
    # P(tk|C) = P(tk|C) = (类C下单词tk在各文档中出现过的次数之和+1)/(类c的单词总数+|V|)
    # 结果最后返回字典，形式为{'类别':[单词频率]}，其中频率保留6位小数点，单词频率列表里的顺序和词库单词顺序一致
    def words_proportion_polynomial(self, words, Lexicon, words_statistics):
        Lexicon_proportion = {}
        spam_proportion = ones(len(Lexicon))
        ham_proportion = ones(len(Lexicon))
        Lexicon_proportion.setdefault('spam',spam_proportion)
        Lexicon_proportion.setdefault('ham',ham_proportion)
        for catagory, c_value in words.items():
            for _, t_words in c_value.items():
                for w in t_words:
                    Lexicon_proportion[catagory][Lexicon.index(w)]+=1
        Lexicon_proportion['spam'] = Lexicon_proportion['spam'] / (words_statistics['spam']+len(Lexicon))
        Lexicon_proportion['ham'] = Lexicon_proportion['ham'] / (words_statistics['ham']+len(Lexicon))
        Lexicon_proportion['spam'] = [round(x,6) for x in Lexicon_proportion['spam']]
        Lexicon_proportion['ham'] = [round(x,6) for x in Lexicon_proportion['ham']]
        return Lexicon_proportion

    # 用于计算单词出现次数
    def words_statistics_polynomial(self, words):
        words_statistics = {}
        words_statistics['all'] = 0
        words_statistics['spam'] = 0
        words_statistics['ham'] = 0
        for catagory, c_value in words.items():
            for _, t_words in c_value.items():
                words_statistics[catagory] += len(t_words)
                words_statistics['all'] += len(t_words)
        return words_statistics

    # 多项式模型下的测试，步骤为：
    # 1、测试集单词分割，与训练集一致
    # 2、进行预测   ---> 调用classify_polynomial函数
    def test_word_polynomial(self, P_S, P_H, Lexicon_proportion, Lexicon):
        # test_path = 'test/'
        # test_words = self.word_cut(test_path, False)
        right = 0
        ham_right = 0
        wrong = 0
        ham_wrong = 0
        wrong_file = []
        for catagory, c_value in self.test_words.items():
            for txt, t_words in c_value.items():
                # print("测试文件为："+txt,end=" ")
                classfiy = self.classify_polynomial(t_words, P_S, P_H, Lexicon_proportion, Lexicon)
                if classfiy == catagory:
                    # print("√√√√√√√√√√判断成功√√√√√√√√√√")
                    right += 1
                    if catagory == "ham":
                        ham_right += 1
                else:
                    # print("！！！！！!判断失败! ！！！！！")
                    wrong += 1
                    if catagory == "ham":
                        ham_wrong += 1
                    wrong_file.append(catagory+txt)
                    if platform.system()=='Windows':
                        self.wrong_file_times["poly"][catagory][int(txt.split('.')[0].split('\\')[-1])-1]+=1
                    elif(platform.system()=='Linux'):
                        self.wrong_file_times["poly"][catagory][int(txt.split('.')[0].split('/')[-1])-1]+=1
        
        self.save_result('poly', wrong, right, ham_wrong, ham_right, wrong_file)

    # 多项式模型下进行预测，步骤为：
    # 1、公式为：P(C|W1,W2...Wn) = P(C)P(W1,W2...Wn|C) = P(C)P(W1|C)P(W2|C)...P(Wn|C)，其中W1到Wn为进行测试的文档中出现的单词
    # 2、对正常邮件和垃圾邮件的概率分别进行计算，最后谁的概率大即为谁
    def classify_polynomial(self, words_list, P_H, P_S, Lexicon_proportion, Lexicon):
        P_SH_W = {}
        P_SH_W.setdefault('spam',P_S)
        P_SH_W.setdefault('ham',P_H)
        for catagory in self.catagory:
            for w in words_list:
                if w in Lexicon:
                    # print(w)
                    P_SH_W[catagory] = P_SH_W[catagory]*Lexicon_proportion[catagory][Lexicon.index(w)]
        # print(P_SH_W)
        if P_SH_W['spam'] > P_SH_W['ham']:
            # print("为正常邮件")
            return 'spam'
        else:
            # print("为垃圾邮件")
            return 'ham'

    def SVM(self):
        # words, self.train_Lexicon = self.word_cut(self.train_file_path)
        train_feature_result, train_feature_label = self.word_feature_svm(self.train_words, self.train_Lexicon)
        svm = self.train_svm(train_feature_result, train_feature_label)
        self.test_svm(svm, self.train_Lexicon)
        
    def word_feature_svm(self, words, Lexicon, test=False):
        feature_result = []
        feature_label = []
        feature_txt = []
        for catagory, c_value in words.items():
            for txt, t_words in c_value.items():
                feature = numpy.zeros(len(Lexicon))
                if catagory == "ham":
                    feature_label.append(1)
                else:
                    feature_label.append(0)
                if test:
                    feature_txt.append(txt)
                for w in t_words:
                    if w in Lexicon:
                        feature[Lexicon.index(w)] += 1
                feature_result.append(feature)
        if test:
            return feature_result, feature_label, feature_txt
        else:
            return feature_result, feature_label

    def train_svm(self, train_feature_result, train_feature_label, save=False):
        svm = LinearSVC()
        svm.fit(train_feature_result, train_feature_label)
        if save:
            joblib.dump(svm,'svm_model.m')
        return svm

    def test_svm(self, svm, Lexicon, model=False):
        if model:
            svm = joblib.load('svm_model.m')
        # test_words = self.word_cut("test/",False)
        test_feature_result, test_feature_label, test_feature_txt = self.word_feature_svm(self.test_words, Lexicon, True)
        result = svm.predict(test_feature_result)

        right = 0
        ham_right = 0
        wrong = 0
        ham_wrong = 0
        wrong_file = []
        for i in range(10):
            if result[i] == test_feature_label[i]:
                # print("√√√√√√√√√√判断成功√√√√√√√√√√")
                right += 1
                if test_feature_label[i] == 1:
                    ham_right += 1
            else:
                # print("！！！！！!判断失败! ！！！！！")
                wrong += 1
                if test_feature_label[i] == 1:
                    ham_wrong += 1
                catagory = "ham" if test_feature_label[i]==1 else "spam"
                wrong_file.append(catagory+test_feature_txt[i])
                if platform.system()=='Windows':
                    self.wrong_file_times["svm"][catagory][int(test_feature_txt[i].split('.')[0].split('\\')[-1])-1]+=1
                elif(platform.system()=='Linux'):
                    self.wrong_file_times["svm"][catagory][int(test_feature_txt[i].split('.')[0].split('/')[-1])-1]+=1

        self.save_result('svm', wrong, right, ham_wrong, ham_right, wrong_file)

    # 获取所有的邮件
    def get_all_file(self):
        self.all_file.setdefault('ham',[])
        self.all_file.setdefault('spam',[])
        for root, dirs, _ in os.walk(self.train_file_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                files = os.listdir(dir_path)
                for file in files:
                    if dir=='ham':
                        self.all_file['ham'].append(os.path.join(dir_path, file))
                    else:
                        self.all_file['spam'].append(os.path.join(dir_path, file))

    # 创建测试文档，从总体的文件中（50份）中正负各选择5份作为测试集
    # 训练文件路径和测试文件路径将保存在self.operation_file中，字典形式{'train':{'spam':[],'ham':[]},'test':{'spam':[],'ham':[]}}
    # 如果要使用自己创建的训练数据集和测试数据集，则自行创建test文件夹，创建spam和ham子文件夹，并将相应的训练文件移入其中，并在命令行传入相应的参数
    def create_test_file(self, aim=False):
        if aim:         # 自行创建训练数据集
            for root, dirs, _ in os.walk(self.train_file_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    files = os.listdir(dir_path)
                    for file in files:
                        if dir=='ham':
                            self.all_file['ham'].append(os.path.join(dir_path, file))
                        else:
                            self.all_file['spam'].append(os.path.join(dir_path, file))
            if not self.test_file_path:
                print("您尚未传入测试集路径")
            for root, dirs, _ in os.walk(self.test_file_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    files = os.listdir(dir_path)
                    for file in files:
                        if dir=='ham':
                            self.all_file['ham'].append(os.path.join(dir_path, file))
                        else:
                            self.all_file['spam'].append(os.path.join(dir_path, file))
        else:
            train_spam_file = self.all_file['spam'][:]
            train_ham_file = self.all_file['ham'][:]

            test_file = []
            
            num = range(0,len(train_ham_file)-1)
            nums = random.sample(num, self.test_file_num)
            for n in nums:
                if platform.system()=='Windows':
                    test_file.append("ham"+train_ham_file[int(n)].split('\\')[-1])
                elif(platform.system()=='Linux'):
                    test_file.append("ham"+train_ham_file[int(n)].split('/')[-1])
                self.operation_file['test']['ham'].append(train_ham_file[int(n)])
            for f in self.operation_file['test']['ham']:
                train_ham_file.remove(f)

            num = range(0,len(train_spam_file)-1)
            nums = random.sample(num, self.test_file_num)
            for n in nums:
                if platform.system()=='Windows':
                    test_file.append("spam"+train_spam_file[int(n)].split('\\')[-1])
                elif(platform.system()=='Linux'):
                    test_file.append("spam"+train_spam_file[int(n)].split('/')[-1])
                self.operation_file['test']['spam'].append(train_spam_file[int(n)])
            for f in self.operation_file['test']['spam']:
                train_spam_file.remove(f)
            self.wrong_file["第 "+str(self.turn)+" 次测试"].setdefault("all", test_file)

            self.operation_file['train']['spam'] = train_spam_file
            self.operation_file['train']['ham'] = train_ham_file

    # 将测试好了的文件进行归为，重新放到训练集中以便下一次再抽取
    def file_recover(self):
        self.operation_file['train']['ham'].clear()
        self.operation_file['train']['spam'].clear()
        self.operation_file['test']['ham'].clear()
        self.operation_file['test']['spam'].clear()


    def result_save_init(self):
        
        self.operation_file.setdefault('train',{}).setdefault('ham',[])
        self.operation_file.setdefault('train',{}).setdefault('spam',[])
        self.operation_file.setdefault('test',{}).setdefault('ham',[])
        self.operation_file.setdefault('test',{}).setdefault('spam',[])
        self.rate.setdefault("paul",{}).setdefault("right",[])
        self.rate.setdefault("paul",{}).setdefault("wrong",[])
        self.rate.setdefault("paul",{}).setdefault("ham-right",[])
        self.rate.setdefault("paul",{}).setdefault("ham-wrong",[])
        self.rate.setdefault("paul",{}).setdefault("spam-right",[])
        self.rate.setdefault("paul",{}).setdefault("spam-wrong",[])
        self.rate.setdefault("poly",{}).setdefault("right",[])
        self.rate.setdefault("poly",{}).setdefault("wrong",[])
        self.rate.setdefault("poly",{}).setdefault("ham-right",[])
        self.rate.setdefault("poly",{}).setdefault("ham-wrong",[])
        self.rate.setdefault("poly",{}).setdefault("spam-right",[])
        self.rate.setdefault("poly",{}).setdefault("spam-wrong",[])
        self.rate.setdefault("svm",{}).setdefault("right",[])
        self.rate.setdefault("svm",{}).setdefault("wrong",[])
        self.rate.setdefault("svm",{}).setdefault("ham-right",[])
        self.rate.setdefault("svm",{}).setdefault("ham-wrong",[])
        self.rate.setdefault("svm",{}).setdefault("spam-right",[])
        self.rate.setdefault("svm",{}).setdefault("spam-wrong",[])
        self.wrong_file_times.setdefault("paul",{}).setdefault("spam",zeros(25))
        self.wrong_file_times.setdefault("paul",{}).setdefault("ham",zeros(25))
        self.wrong_file_times.setdefault("poly",{}).setdefault("spam",zeros(25))
        self.wrong_file_times.setdefault("poly",{}).setdefault("ham",zeros(25))
        self.wrong_file_times.setdefault("svm",{}).setdefault("spam",zeros(25))
        self.wrong_file_times.setdefault("svm",{}).setdefault("ham",zeros(25))
        self.wrong_num.setdefault("paul",[])
        self.wrong_num.setdefault("poly",[])
        self.wrong_num.setdefault("svm",[])

    def draw_result(self):
        test_times = [str(i+1) for i in range(self.test_times)]
        line = Line()
        line.add_xaxis(test_times)
        if self.model_choose==1 or self.model_choose==4 or self.model_choose==5:
            line.add_yaxis(
                "伯努利",
                self.rate["paul"]["right"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        if self.model_choose==2 or self.model_choose==4 or self.model_choose==5:
            line.add_yaxis(
                "多项式",
                self.rate["poly"]["right"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        if self.model_choose==3 or self.model_choose==4 or self.model_choose==5:
            line.add_yaxis(
                "SVM",
                self.rate["svm"]["right"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        line.set_global_opts(title_opts=opts.TitleOpts(title="正确率"))
        line.render("正确率.html")

        line_wrong = Line()
        line_wrong.add_xaxis(test_times)
        if self.model_choose==1 or self.model_choose==4 or self.model_choose==5:
            line_wrong.add_yaxis(
                "伯努利",
                # self.rate["paul"]["spam-wrong"],
                self.wrong_num["paul"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        if self.model_choose==2 or self.model_choose==4 or self.model_choose==5:
            line_wrong.add_yaxis(
                "多项式",
                # self.rate["poly"]["spam-wrong"],
                self.wrong_num["poly"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        if self.model_choose==3 or self.model_choose==4 or self.model_choose==5:
            line_wrong.add_yaxis(
                "SVM",
                # self.rate["svm"]["spam-wrong"],
                self.wrong_num["svm"],
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
        line_wrong.set_global_opts(title_opts=opts.TitleOpts(title="错误邮件个数"))
        line_wrong.render("错误邮件个数.html")

        paul_pie = Pie()
        poly_pie = Pie()
        svm_pie = Pie()
        spam_test_file = ['spam_'+str(i+1) for i in range(25)]
        if self.model_choose==1 or self.model_choose==4:
            paul_spam_test_file = [list(z) for z in zip(spam_test_file, self.wrong_file_times["paul"]["spam"])]
            paul_spam_test_file = [i for i in paul_spam_test_file if i[1]!=0]
            paul_spam_test_file.sort(key=lambda x: x[1])
            paul_pie.add(
                "", 
                paul_spam_test_file,
                center=["25%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )
        if self.model_choose==2 or self.model_choose==4 or self.model_choose==5:
            poly_spam_test_file = [list(z) for z in zip(spam_test_file, self.wrong_file_times["poly"]["spam"])]
            poly_spam_test_file = [i for i in poly_spam_test_file if i[1]!=0]
            poly_spam_test_file.sort(key=lambda x: x[1])
            poly_pie.add(
                "", 
                poly_spam_test_file,
                center=["25%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )
        if self.model_choose==3 or self.model_choose==4 or self.model_choose==5:
            svm_spam_test_file = [list(z) for z in zip(spam_test_file, self.wrong_file_times["svm"]["spam"])]
            svm_spam_test_file = [i for i in svm_spam_test_file if i[1]!=0]
            svm_spam_test_file.sort(key=lambda x: x[1])
            svm_pie.add(
                "", 
                svm_spam_test_file,
                center=["25%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )

        ham_test_file = ['ham_'+str(i+1) for i in range(25)]
        
        if self.model_choose==1 or self.model_choose==4 or self.model_choose==5:
            paul_ham_test_file = [list(z) for z in zip(ham_test_file, self.wrong_file_times["paul"]["ham"])]
            paul_ham_test_file = [i for i in paul_ham_test_file if i[1]!=0]
            paul_ham_test_file.sort(key=lambda x: x[1])
            paul_pie.add(
                "", 
                paul_ham_test_file,
                center=["70%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )
        if self.model_choose==2 or self.model_choose==4 or self.model_choose==5:
            poly_ham_test_file = [list(z) for z in zip(ham_test_file, self.wrong_file_times["poly"]["ham"])]
            poly_ham_test_file = [i for i in poly_ham_test_file if i[1]!=0]
            poly_ham_test_file.sort(key=lambda x: x[1])
            poly_pie.add(
                "", 
                poly_ham_test_file,
                center=["70%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )
        if self.model_choose==3 or self.model_choose==4 or self.model_choose==5:
            svm_ham_test_file = [list(z) for z in zip(ham_test_file, self.wrong_file_times["svm"]["ham"])]
            svm_ham_test_file = [i for i in svm_ham_test_file if i[1]!=0]
            svm_ham_test_file.sort(key=lambda x: x[1])
            svm_pie.add(
                "", 
                svm_ham_test_file,
                center=["70%", "50%"],
                radius=[0, 100],
                label_opts=self.new_label_opts(),
            )
        paul_pie.set_global_opts(
            title_opts=opts.TitleOpts(title="伯努利模型-邮件错误识别"),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top="10%", pos_left="90%", orient="vertical")
        )
        paul_pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        paul_pie.render("伯努利模型-邮件错误识别.html")

        poly_pie.set_global_opts(
            title_opts=opts.TitleOpts(title="多项式模型-邮件错误识别"),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top="10%", pos_left="90%", orient="vertical")
        )
        poly_pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        poly_pie.render("多项式模型-邮件错误识别.html")

        svm_pie.set_global_opts(
            title_opts=opts.TitleOpts(title="SVM-邮件错误识别"),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top="10%", pos_left="90%", orient="vertical")
        )
        svm_pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        svm_pie.render("SVM-邮件错误识别.html")

        make_snapshot(driver, line.render(), "正确率.png")
        print("正确率图片生成完毕")
        make_snapshot(driver, line.render(), "正确率.png")
        print("错误邮件个数图片生成完毕")
        make_snapshot(driver, paul_pie.render(), "伯努利模型-邮件错误识别.png")
        print("伯努利模型-邮件错误识别图片生成完毕")
        make_snapshot(driver, poly_pie.render(), "多项式模型-邮件错误识别.png")
        print("多项式模型-邮件错误识别图片生成完毕")
        make_snapshot(driver, svm_pie.render(), "SVM-邮件错误识别.png")
        print("SVM-邮件错误识别图片生成完毕")

    def new_label_opts(self):
        fn = """
            function(params) {
                if(params.name == '其他')
                    return '\\n\\n\\n' + params.name + ' : ' + params.value + '%';
                return params.name + ' : ' + params.value + '%';
            }
            """
        return opts.LabelOpts(formatter=JsCode(fn), position="center")


parser = argparse.ArgumentParser()
parser.add_argument("--model",type=int,default=1,help="模型选择：伯努利--1  多项式--2  SVM--3  三者同时测试且自定了数据集--4  三者同时测试且随机生成数据集--5",choices=[1,2,3,4,5,6])
parser.add_argument("--train_path",type=str,default="train",help="训练集路径")
parser.add_argument("--test_path",type=str,default="test",help="测试集路径")
parser.add_argument("--test_times",type=int,default=1,help="测试次数")
parser.add_argument("--test_file_num",type=int,default=5,help="每一类测试集的数目")
args = parser.parse_args()

bayesian = Bayesian(args)
bayesian.bayes()
bayesian.draw_result()
# bayesian.create_test_file()
# bayesian.word_cut()
# bayesian.bayes_paul()
# bayesian.bayes_polynomial()
# bayesian.file_recover()

