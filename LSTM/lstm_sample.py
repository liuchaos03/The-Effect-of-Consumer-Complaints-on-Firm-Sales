# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:16:17 2021

@author: zheng xin
"""
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy
import sklearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import xlwt
import numpy as np 
import tensorflow

def lstmsimple(MAX_NB_WORDS,EMBEDDING_DIM,X,Y,cat_id_df):
    #-----------设置模型
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
 #   model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(120, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
  #  model.add(LSTM(20, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(20))
    model.add(Dense(len(cat_id_df), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    # sparse_categorical_crossentropy   categorical_crossentropy
    print(model.summary())
    
    #-----------执行
    epochs = 12
    batch_size =10 # 10
    history = model.fit(X, Y, epochs=epochs, shuffle=False,batch_size=batch_size,validation_split=0.3,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    
    
    #----------输出结果-------------
    
    
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show();
    
    
    #----------输出结果-------------
    y_preda = model.predict(X)
    y_pred = y_preda.argmax(axis = 1)
    Y_test = Y.argmax(axis = 1)
    key=y_pred==Y_test;
    Acc=sum(key)/len(y_pred)
    print('基础模型准确率：%4.8f' % Acc)
    return model,y_preda,key,Acc

def Dim_data(X,Y,D):
    key=numpy.where(D==False)
    key=list(key[0])
    X1=numpy.append(X,X[key,:],axis=0)
    X1=numpy.append(X1,X[key,:],axis=0)
    
    Y1=numpy.append(Y,Y[key,:],axis=0)
    Y1=numpy.append(Y1,Y[key,:],axis=0)
    return X1,Y1
    


##-------------
tensorflow.random.set_seed(2337)  #2337
tasks='sample.csv'
path=r'F:\竞赛与项目\项目2020\论文10-txt\deep\\'

df = pd.read_csv(path+tasks, encoding='ANSI')
df = sklearn.utils.shuffle(df, random_state=10)  #打乱顺序
df=df[['obj','review']]

#----清洗空的文本
df[df.isnull().values==True]
df = df[pd.notnull(df['review'])]

#------标签编码分类
df['id'] = df['obj'].factorize()[0]
cat_id_df = df[['obj', 'id']].drop_duplicates().sort_values('id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['id', 'obj']].values)

#df['review']=df['review'].values
#df.sample(10)
# 设置最频繁使用的20000个词
MAX_NB_WORDS = 50
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 20
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 5
 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['review'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))


X=tokenizer.texts_to_sequences(df['review'].values)
#填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#多类标签的onehot展开
Y = pd.get_dummies(df['id']).values
print("样本量：",X.shape,Y.shape)


#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size = 0.1, random_state = 42)
#Acc=1.0;
#while Acc==1.0:    #直到
X_train=X;
Y_train=Y;

print("训练量：",X_train.shape,Y_train.shape)


##----------训练多少个基础模型-----------
D=numpy.zeros(len(Y))
D=D==D
Adaboost=numpy.zeros((len(Y),len(cat_to_id)));

for i in range(0,1):
    X_train,Y_train=Dim_data(X_train,Y_train,D);
    model,y_preda,D,Acc=lstmsimple(MAX_NB_WORDS,EMBEDDING_DIM,X_train,Y_train,cat_id_df)
    Adaboost=y_preda[0:len(Y),:]*Acc+Adaboost
  
Y_pred = Adaboost.argmax(axis = 1)
Y_targ=Y.argmax(axis = 1)
Acc=sum(Y_pred==Y_targ)/len(Y_targ)
print('Adaboost准确率：%4.8f' % Acc)
#生成混淆矩阵
conf_mat = confusion_matrix(Y_targ, Y_pred)
labels=list(cat_id_df['obj'])

workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('confusion')

worksheet.write(0,0, label = '')
for i in range(len(labels)): 
    worksheet.write(i+1,0, label =labels[i])
    worksheet.write(0,i+1, label =labels[i])

for i in range(len(labels)):
    for j in range(len(labels)):
        worksheet.write(i+1,j+1, int(conf_mat[i][j]))
        
workbook.save(path+tasks+'_confusion_matrix'+'.xls')
