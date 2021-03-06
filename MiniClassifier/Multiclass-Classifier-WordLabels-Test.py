# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:30:41 2020

@author: Abhilash
"""

import MiniClassifier as mc
from sklearn.model_selection import train_test_split

import pandas as pd

if __name__=='__main__':
    '''
    This contains a script for testing performance of the library on binary classification corpus.
    Here the labels are in the form of text ,and hence we have to convert these textual labels into 
    numeric ones by using Label Encoder. If the textual labels contain space in between them, then
    that space is removed for the encoding to operate.
    The data is abstracted from Kaggle -BBC News Classification Contest(https://www.kaggle.com/yufengdev/bbc-fulltext-and-category)
    The workflow should remain the same for different classification problems (supervised deep learning).
    For numeric labels/targets, there is no  manipulation required to be done and is provided in the 
    other Test script(Binary classifier).
    In this case, we will be tuning the hyperparameters so as to accomodate multiclass/categorical logits. For this, 
    we change the 'final_activation' parameter to be softmax. The rest of the parameters can be left
    as is.
    There is a scope of involving  Pretrained Embeddings.For this the library requires the relative 
    path of the embedding file ,for example: Glove840B.200D or word2vec ,etc and these files are generally
    in .txt or other formats.Recommended to download these from Kaggle (available in different formats).
    
    While using Pre-trained embeddings, there are 2 things that are to be done. In this example, in the 
    case of bilstm model, the "bilstm.parameters" method contains certain arguements which builds the model.
    The last and the second last arguement are important: the last arguement contains the relative path to 
    the embedding file and the second last arguement specifies whether pretrained embeddings are used or not.
    If pretrained embeddings are used the parameters should be True,<path to embedding>.
    
    If pretrained embeddings are not used then, the arguements can be set to False,<path to anything>.
    This allows the library to use default keras Embeddings at runtime.
    
    This module shows the 3 different architectures which are present in the library- 
    1. Bilstm Model
    2. Simple Dense -LSTM model
    3. Convolution Dense Model
    
    Anyone can be used as per usecase, while here the 3 different architectures are shown together for proper
    usability and understandability.
    The workflow is as follows:
        
    1. Read in the data frame (pandas dataframe)
    2. Segregate the input text corpus(preferably 'X') from the target labels('Y')
    3. Here 'Y' is textual so  LabelEncoder will encode it for our usecase. If the labels are space
       separated, then that space has to be removed before label encoding.
    4. The tokenization and padding of the 'X' (text corpus) should be done as provided.
       The dataset should be split into test and training sets using sklearn.
    5. The Tokenizer is important for the model as  it is used in evaluating the pretrained embedding.
    6. The next step is to specify the hyperparameters such as  dense_units,lstm_units, etc.
    7. Depending on the use case, any one of the 3 models can be chosen
    8. For each of the models, the sequence should be :
        8.1 Initialize the object of the Neural Network class - such as BiLSTM_Cell()
        8.2 Specify the parameters using the <modelname>.parameters() method with the arguements
        8.3 The <modelname>.build_bilstm_neuron() method builds the network.
        8.4 The <modelname>.fit_model and <modelname>.evaluate methods are used for training and evaluating
    9. This pattern is same for all the 3 models.
    10. The Predictor class is used for predicting the outputs from the test dataset.
    
    
    The entire codebase is a workflow and can be used as is, with changes in the dataframe and
    the hyperparameters.
    '''
    
    '''
    Load dataframe and extract the text corpus(x) and the target labels (y)
    '''
    df=pd.read_csv("D:\\archive\\bbc-text.csv")
    #df.head()
    X=(df['text'])
    li=df['category']
    '''Remove spaces from the labels /targets if textual labels are used.'''
    for j in range(len(li)):
        li[j]= "".join(str(li[j]).split())
    #print(li)
    Y=(li)
    #print(Y.shape)
    '''Encode the labels using Label Encoder.'''
    Y_unique=list(set(li))
    '''Hyperparameter (maxwords, maxlen) for Maximum words in Embedding and Maximum length for the sentence after which
    padding is done'''
   
    maxwords=1000
    maxlen=500
    '''Creating the tokenizer object and creating padded tokens from the text corpus (X).
    This calls the SimpleTokenizer Class.'''
    
    token=mc.Simple_tokenizer(maxwords,maxlen,X)
    tokeni=token.create_tokenizer()
    encoded_token=token.encode_tokenizer(X)
    pad_token,vocab_size=token.padded_tokenizer()
    print("padded token".format(),pad_token)
    print("vocab_size".format(),vocab_size)
    print('Padded Token Shape'.format(),pad_token.shape)
    #print(Y_unique)
    labels,encoder=token.labelencode_labels(Y)
    print('Labels shape'.format(),labels.shape)
    '''Split the dataset into test and train using sklearn'''
    X_train,X_test,Y_train,Y_test= train_test_split(pad_token,labels,test_size=0.2)
    print('X_train shape'.format(),X_train.shape)
    print('Y_train shape'.format(),Y_train.shape)
    tokenizer=tokeni
    

    '''Set the hyperparameters for the models.Names are self-explanatory'''
    print("==============================")
    output_samples=labels.shape[-1]
    embedding_dim=512
    dense_units=64
    lstm_units=64
    bilstm_units=64
    activation='relu'
    final_activation='softmax'
    optimizer='adam'
    training_epochs=5
    training_batch_size=150
    val_epochs=5
    val_batch_size=150
    filter_size=128
    kernel_size=5
    #Specify path to the Pretrained Embedding file
    path='D:\\glove.6B.200d\\glove.6B.200d.txt'
    #If no pretrained embedding is required.For this the second last arguement in parameters method should be False
    path1=''
    
    '''Demonstration for the Bilstm model.The parameters are self explanatory and used from the hyperameters set.'''
    print("Bilstm model for Evaluation")
    bilstm=mc.BiLSTM_Cell()
    #if no pretrained embedding
    #bilstm.parameters(activation,final_activation,embedding_dim,dense_units,bilstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,False,path1)
    #Use pretrained glove embdding
    bilstm.parameters(activation,final_activation,embedding_dim,dense_units,bilstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,True,path)
    bilstm.build_bilstm_neuron(maxwords,maxlen)
    print('bilstm X_train shape'.format(),bilstm.X_train.shape)
    print('bilstm Y_train shape'.format(),bilstm.Y_train.shape)
    print("Evualating the Model-Training")
    bilstm.fit_model(training_epochs,training_batch_size)
    print("Evualating the Model-Validation")
    bilstm.evaluate(val_epochs,val_batch_size)
    print("Prediction of Labels")
    #Predictor class for predicting
    predictor=mc.Predictor()
    pred_list=predictor.predict(bilstm.model,bilstm.X_test,X,Y,encoder)
    print("=====================================================")
         
    '''Demonstration for the Dense model.The parameters are self explanatory and used from the hyperameters set.'''
    print("Dense model for Evualation")
    dense=mc.Dense_Cell()
    dense.parameters(activation,final_activation,embedding_dim,dense_units,lstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,True,path)
    dense.build_dense_neuron(maxwords,maxlen)
    print('Dense X_train shape'.format(),dense.X_train.shape)
    print('Dense Y_train shape'.format(),dense.Y_train.shape)
    print("Evualating the Model-Training")
    dense.fit_model(training_epochs,training_batch_size)
    print("Evualating the Model-Validation")
    dense.evaluate(val_epochs,val_batch_size)
    #Predictor class for predicting
    predictor=mc.Predictor()
    pred_list=predictor.predict(dense.model,dense.X_test,X,Y,encoder)
        
    print("=====================================================")
    
    '''Demonstration for the Convolution model.The parameters are self explanatory and used from the hyperameters set.'''
    conv=mc.Convolution_Cell()
    conv.parameters(activation,final_activation,embedding_dim,filter_size,kernel_size,dense_units,lstm_units,output_samples,optimizer,X_train,Y_train,X_test,Y_test,tokenizer,False,path)
    conv.build_conv1d_neuron(maxwords,maxlen)
    print('Convolution X_train shape'.format(),conv.X_train.shape)
    print('Convolution Y_train shape'.format(),conv.Y_train.shape)
    print("Evualating the Model-Training")
    conv.fit_model(training_epochs,training_batch_size)
    print("Evualating the Model-Validation")
    conv.evaluate(val_epochs,val_batch_size)
    #Predictor class for predicting
    predictor=mc.Predictor()
    pred_list=predictor.predict(conv.model,conv.X_test,X,Y,encoder)
        
    print("=====================================================")
    
    
                