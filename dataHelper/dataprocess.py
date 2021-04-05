import os
import re
from keras.preprocessing import sequence  #用于进行取长补短操作，将长度统一设定为100
from keras.preprocessing.text import Tokenizer #用于创建token词典

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

def read_files(filetype):
    path = "../data/aclImdb_v1/aclImdb/"
    file_list = []

    positive_path = path + filetype +"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path = path + filetype +"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]

    all_labels = ([1]*12500+[0]*12500)

    all_texts = []
    for fi in file_list:
        with open(fi,encoding='utf-8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels,all_texts


if __name__ == '__main__':
    y_train,train_text = read_files("train")
    y_test,test_text = read_files("test")

    token = Tokenizer(num_words=2000)#建立一个有2000个单词的字典
    token.fit_on_texts(train_text)#读取所有的影评，按照单词出现的频数进行排序，取前2000个形成字典

    x_train_seq = token.texts_to_sequences(train_text)#将训练和测试的文章转换为数字列表，以便以后进行计算
    x_test_seq = token.texts_to_sequences(test_text)

    x_train = sequence.pad_sequences(x_train_seq,maxlen=100)#利用方法进行取长补短操作
    x_test = sequence.pad_sequences(x_test_seq,maxlen=100)

    print(x_train[0])
    print(type(x_train))
    ''' 
    测试数据正确
    查看评价信息
    print(train_text[0])
    print(y_train[0])
    print(test_text[12500])
    print(y_test[12500])
    文章转换成数字列表
    print(train_text[0])
    print(x_train_seq[0])
    
    查看读取了多少文章
    print(token.document_count)
    查看统计后的index和数据对应情况
    print(token.word_index)
    
    print('before pad sequence length=',len(x_train_seq[0]))
    print(x_train_seq[0])
    print('after pad sequence length=',len(x_train[0]))
    print(x_train[0])
    '''

