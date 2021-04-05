from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import os
np.random.seed(10)

class Data:
    def __init__(self,train_set,train_label,test_set,test_label,train_seq,test_seq,train,test):
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.train = train
        self.test = test


    def rm_tags(self,text):
        re_tag = re.compile(r'<[^>]+>')
        return re_tag.sub('',text)

    def read_files(self,filetype):
        path = "../data/aclImdb_v1/aclImdb/"
        file_list = []

        positive_path = path + filetype + "/pos/"
        for f in os.listdir(positive_path):
            file_list += [positive_path + f]

        negative_path = path + filetype + "/neg/"
        for f in os.listdir(negative_path):
            file_list += [negative_path + f]

        all_labels = ([1] * 12500 + [0] * 12500)

        all_texts = []
        for fi in file_list:
            with open(fi, encoding='utf-8') as file_input:
                all_texts += [self.rm_tags(" ".join(file_input.readlines()))]

        return all_labels, all_texts

    def data_process(self):
        self.train_label,self.train_set = self.read_files("train")
        self.test_label,self.test_set = self.read_files("test")

        token = Tokenizer(num_words=2000)
        token.fit_on_texts(self.train_set)

        self.train_seq = token.texts_to_sequences(self.train_set)
        self.test_seq = token.texts_to_sequences(self.test_set)

        self.train = sequence.pad_sequences(self.train_seq,maxlen=100)
        self.test = sequence.pad_sequences(self.test_seq,maxlen=100)


    def me(self):
        print(self.train[0])




