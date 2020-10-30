import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense,Flatten,Conv2D,Conv1D,GlobalMaxPooling1D
from keras.optimizers import Adam
import numpy as np  
import pandas as pd            
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt

class Simple_tokenizer():
    def __init__(self,maxwords,maxlen,sample):
        self.maxwords=maxwords
        self.maxlen=maxlen
        self.sample=sample
        super(Simple_tokenizer,self).__init__()
        
    def create_tokenizer(self):
        self.tokenizer=Tokenizer(num_words=self.maxwords)
        self.tokenizer.fit_on_texts(self.sample)
        return self.tokenizer
    def encode_tokenizer(self,input_words):
        self.input_words=input_words
        self.encoded_text=self.tokenizer.texts_to_sequences(self.input_words)
        return self.encoded_text
    def padded_tokenizer(self):
        self.padded_text= pad_sequences(self.encoded_text,maxlen=self.maxlen,padding='post')
        self.vocab_size=len(self.tokenizer.word_index)+1
        return self.padded_text,self.vocab_size
    def labelencode_labels(self,sample):
        self.sample=sample
        self.encoder = LabelEncoder()
        self.encoder.fit(self.sample)
        self.sample = self.encoder.transform(self.sample)
        self.num_classes = np.max(self.sample) + 1
        self.labels = keras.utils.to_categorical(self.sample, self.num_classes)
        return self.labels,self.encoder

    def show_tokenized_words(self):
        self.tokenized_words=self.tokenizer.word_index
        return self.tokenized_words 
class Dense_Cell():
    def parameters(self,activation,final_activation,embedding_dim,dense_units,lstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,use_pretrained,path):
        self.activation=activation
        self.embedding_dim=embedding_dim
        self.dense_units=dense_units
        self.output_samples=output_samples
        self.final_activation=final_activation
        self.lstm_units=lstm_units
        self.optimizer=optimizer
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.tokenizer=tokenizer
        self.use_pretrained=use_pretrained
        self.path=path
    def build_dense_neuron(self,maxwords,maxlen):
        
        self.maxwords=maxwords
        self.maxlen=maxlen
        self.model=Sequential()
        if (self.use_pretrained==True):
            pretrained_embed=Pretrained_Embedding()
            self.embedding_vector,self.embedding_matrix,self.embed_size=pretrained_embed.load_embedding(self.maxwords,self.tokenizer,self.path)
            self.model.add(Embedding(self.maxwords,self.embed_size,weights=[self.embedding_matrix]))
        else:
            self.model.add(Embedding(self.maxwords,self.embedding_dim,input_length=self.maxlen))
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dense(self.dense_units,activation=self.activation))
        self.model.add(Dense(self.output_samples,activation=self.final_activation))
        self.model.compile(loss='categorical_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])
        self.model.summary()
        return self.model
    def fit_model(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_train,self.Y_train,self.batch_size,self.epochs,verbose=2,validation_split=0.1)
    def evaluate(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_test,self.Y_test,self.batch_size,self.epochs,verbose=2)
    
    
class BiLSTM_Cell():
    def parameters(self,activation,final_activation,embedding_dim,dense_units,bilstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,use_pretrained,path):
        self.activation=activation
        self.final_activation=final_activation
        self.embedding_dim=embedding_dim
        self.dense_units=dense_units
        self.bilstm_units=bilstm_units
        self.output_samples=output_samples
        self.optimizer=optimizer
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.tokenizer=tokenizer
        self.use_pretrained=use_pretrained
        self.path=path
    def build_bilstm_neuron(self,maxwords,maxlen):
        self.maxwords=maxwords
        self.maxlen=maxlen
        self.model=Sequential()
        if (self.use_pretrained==True):
            pretrained_embed=Pretrained_Embedding()
            self.embedding_vector,self.embedding_matrix,self.embed_size=pretrained_embed.load_embedding(self.maxwords,self.tokenizer,self.path)
            self.model.add(Embedding(self.maxwords,self.embed_size,weights=[self.embedding_matrix]))
        else:
            self.model.add(Embedding(self.maxwords,self.embedding_dim,input_length=self.maxlen))

        self.model.add(Bidirectional(LSTM(self.bilstm_units,return_sequences=True)))
        self.model.add(Bidirectional(LSTM(self.bilstm_units)))
        self.model.add(Dense(self.dense_units,activation=self.activation))
        self.model.add(Dense(self.dense_units,activation=self.activation))
        self.model.add(Dense(self.output_samples,activation=self.final_activation))
        self.model.compile(loss='categorical_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])
        
        self.model.summary()
        return self.model
    def fit_model(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_train,self.Y_train,self.batch_size,self.epochs,verbose=2)
    def evaluate(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_test,self.Y_test,self.batch_size,self.epochs,verbose=2)
    

class Convolution_Cell():
    def parameters(self,activation,final_activation,embedding_dim,filter_size,kernel_size,dense_units,bilstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,use_pretrained,path):
        self.activation=activation
        self.final_activation=final_activation
        self.embedding_dim=embedding_dim
        self.filter_size=filter_size
        self.kernel_size=kernel_size
        self.optimizer=optimizer
        self.output_samples=output_samples
        self.dense_units=dense_units
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.tokenizer=tokenizer
        self.use_pretrained=use_pretrained
        self.path=path
    def build_conv1d_neuron(self,maxwords,maxlen):
        self.maxwords=maxwords
        self.maxlen=maxlen
        self.model=Sequential()
        if (self.use_pretrained==True):
            pretrained_embed=Pretrained_Embedding()
            self.embedding_vector,self.embedding_matrix,self.embed_size=pretrained_embed.load_embedding(self.maxwords,self.tokenizer,self.path)
            self.model.add(Embedding(self.maxwords,self.embed_size,weights=[self.embedding_matrix]))
        else:
            self.model.add(Embedding(self.maxwords,self.embedding_dim,input_length=self.maxlen))

        
        self.model.add(Conv1D(self.filter_size,self.kernel_size,activation=self.activation))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(self.dense_units,activation=self.activation))
        self.model.add(Dense(self.dense_units,activation=self.final_activation))
        self.model.add(Dense(self.output_samples,activation=self.final_activation))
        self.model.compile(loss='categorical_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])
        self.model.summary()
        return self.model
    
    def fit_model(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_train,self.Y_train,self.batch_size,self.epochs,verbose=2,validation_split=0.1)
    def evaluate(self,epochs,batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        self.model.fit(self.X_test,self.Y_test,self.batch_size,self.epochs,verbose=2)
class Predictor():
    def predict(self,model,X_test,X,Y,encoder):
        self.model=model
        self.encoder=encoder
        self.X_test=X_test
        self.X=X
        self.Y=Y
        self.text_labels=self.encoder.classes_
        self.prediction_list=[]
        for j in range(len(self.X_test)):
            self.predictions=self.model.predict(np.array([self.X_test[j]]))
            self.predict_label=self.text_labels[np.argmax(self.predictions)]
            print("====================")
            print("question:".format(),self.X[j][:50])
            print("actual answer:".format(),self.Y[j])
            print("predicted answer:".format(),self.predict_label)
            print("======================")
            self.prediction_list.append(self.predict_label)
        return self.prediction_list
            

        
class Pretrained_Embedding():
    def load_embedding(self,maxwords,tokenizer,path):
        self.maxwords=maxwords
        self.max_features=maxwords
        self.tokenizer=tokenizer
        EMBEDDING_FILE = path
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding='utf-8'))

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        word_index = self.tokenizer.word_index
        nb_words = min(self.maxwords, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector  
        return embedding_vector,embedding_matrix,embed_size
    def plot_embeddings(self,idx):
        self.idx=idx
        if self.idx<self.embedding_matrix.shape[1]:
            plt.plot(self.embedding_matrix[self.idx])
            plt.show()
    
