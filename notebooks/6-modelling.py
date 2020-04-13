#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install git+https://www.github.com/keras-team/keras-contrib.git --user -q')
get_ipython().system(u'pip install spacy --user -q')
get_ipython().system(u'pip install glove_python --user -q')


# In[2]:


from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from glove import Corpus, Glove
import glob
import numpy as np
import pandas as pd
import random
import pickle
from spacy.lang.xx import MultiLanguage
nlp = MultiLanguage() #pre-trained model NER
import math


# In[3]:


def readline(filename):
    """
        read file
        return
        format [ ['segelas', 'TYPES'], ['douni', 'PRODUCT], ['parfum', 'PRODUCT'], ['collection', 'PRODUCT'], ['fusion', 'PRODUCT']]
    """
    sentences = []
    sentence = []
    with open(filename, encoding='utf-8') as f:
        for i in f:
            if(len(i[:-1]) < 2) or i.startswith("-DOCSTART-"):
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = i[:-1].split(",")
            sentence.append([splits[0], splits[-1]])
        
        if len(sentence) > 0:
            sentences.append(sentence)
            sentence = []

    return sentences


# In[4]:


def createMatrices(sentences, word2Idx, label2Idx, char2Idx):
    """
    Get the matrices for every dataset. Different with normal NER, we do not use casing-features here, because the input 
    from ASR would be case insensitive
        sentences(list) : list of list of words from the input.
        word2Idx(dict)   : Word and its index
        label2Idx(dict)   : Label and its index
        char2Idx(dict)   : Characters and its index
        
    Output:
        dataset(matrix)   :Matrix containing all of the features.
    """
    unknownIdx = word2Idx["UNK"]
    paddingIdx = word2Idx["PAD"]
    
    dataset = []
    
    word_count = 0
    unknown_word_count = 0
    
    for sentence in sentences:
        word_indices = []
        char_indices = []
        label_indices = []
        for word, char, label in sentence:
            word_count += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknown_word_count += 1
            
            charIdx = []
            for x in char:
                try:
                    charIdx.append(char2Idx[x])
                except KeyError:
                    charIdx.append(char2Idx["UNK"])
            
            word_indices.append(wordIdx)
            char_indices.append(charIdx)
            label_indices.append(label2Idx[label])
            
        dataset.append([word_indices, char_indices, label_indices])
    
    return dataset


# In[5]:


def addCharInformation(sentences):
    """
    Split the word in the sentences from 1 dataset into list of characters:
        ex: [["S","e","l","a","m","a","t"],["P","a","g","i"]]
    Input:
        sentences(list)   : The dataset
    """    
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [i for i in data[0]]
            sentences[i][j] = [data[0], chars, data[1]] ## data[0] is token, chars is chars of token, data[1] is label/entity
    return sentences


# In[6]:


def padding(sentences):
    """
    Because the batches already with same length, we are not padding it anymore. We padding the characters instead
        ex: [["S","e","l","a","m","a","t"],["P","a","g","i","<PAD>","<PAD>","<PAD>"]]
        
    Input:
        sentence(list)   :The dataset
    """
    
    maxwordlength = 15 #15 character max in 1 word
    for i,sentence in enumerate(sentences):
        sentences[i][1] = pad_sequences(sentences[i][1], maxwordlength, padding='pre',truncating='post')
    return sentences


# In[7]:


def createBatches(data):
    """
    Create the batches for train_on_batch keras.
    Input:
        data(list)   :The dataset
    """
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)  ## kata
    batches = []
    batch_len = []
    z = 0
    
    for i in l:
        for batch in data:
            if(len(batch[0]) == i):
                batches.append(batch)
                z+=1
        batch_len.append(z)
    return batches, batch_len


# In[8]:


def iterate_minibatches(dataset,batch_len):
    """
    Generator for dataset to be yielded during training
    Input:
        Dataset(list) : The dataset
        batch_len(list) :Start and end batch_len
    Output:
        Array of [labels, tokens, casing, char] features
    """
    while True:
        start = 0
        for i in batch_len:
            tokens = []
            char = []
            labels = []
            data = dataset[start:i]
            start = i
            for dt in data:
                t,ch,l = dt
                l = np.expand_dims(l, -1)
                tokens.append(t)
                char.append(ch)
                labels.append(l)
            feature = [np.asarray(tokens),np.asarray(char)]
            yield feature, np.asarray(labels)


# In[9]:


## Predict data
def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, char, labels = data
        tokens = np.asarray([tokens])
        char = np.asarray([char])
        
        pred = model.predict([tokens, char], verbose = False)[0]
        pred = pred.argmax(axis = -1) #Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    
    return predLabels, correctLabels


# # 1. Open the file needed and Convert the word into list of chars

# In[10]:


trainSentences = readline("../data/clean/dataset.csv")
trainSentences = addCharInformation(trainSentences)

testSentences = readline("../data/clean/test.csv")
testSentences = addCharInformation(testSentences)


# # 2. Get the Label from the dataset, and convert it to index

# In[11]:


label_set = set()
words = {}
for sentence in trainSentences:
    for token, char, label in sentence:
        label_set.add(label)
        words[token.lower()] = True

label2idx = {v:k for k,v in enumerate(sorted(label_set))}
idx2label = {v:k for k,v in label2idx.items()}


# # 3. Mittens Word Embedding (Extensions for GloVe model). Its retrofitting model

# In[12]:


## Load word2idx pickle
pickle_word2idx_glove = open("../data/pickle_file/word2idx_glove.pkl","rb")
word2idx_glove = pickle.load(pickle_word2idx_glove)
word2Idx = {i[0]:i[1] for i in word2idx_glove}

## Load mittens embedding
embedding_file_path = '../model/mittens_embedding.pkl'
wordEmbeddings = pickle.load(open(embedding_file_path, "rb"))


# In[13]:


## n-dimensional vector of word embeddings
n_dim = wordEmbeddings.shape[1]

word2Idx.update({'PAD': len(word2Idx)})
wordEmbeddings = np.concatenate((wordEmbeddings, np.zeros((1, n_dim))))
word2Idx.update({'UNK': len(word2Idx)})
wordEmbeddings = np.concatenate((wordEmbeddings, np.random.uniform(-0.25*10**-3, 0.25*10**-3,n_dim).reshape((1,n_dim))))


# # 4. Get list of characters and its index

# In[14]:


char2Idx = {"PAD":0, "UNK":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)


# # 5. Create Dataset

# In[15]:


train_set = createMatrices(trainSentences, 
               word2Idx = word2Idx, 
               label2Idx = label2idx,
               char2Idx = char2Idx)

test_set = createMatrices(testSentences, 
               word2Idx = word2Idx, 
               label2Idx = label2idx,
               char2Idx = char2Idx)


# In[16]:


train_set = padding(train_set)
test_set = padding(test_set)


# In[17]:


X_train, X_test = train_test_split(train_set, train_size = 0.75, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_test, train_size= 0.5, random_state=42, shuffle=True)


# In[18]:


train_batch, train_batch_len = createBatches(X_train)
val_batch, val_batch_len = createBatches(X_val)
test_batch, test_batch_len = createBatches(X_test)
real_test_batch, real_test_batch_len = createBatches(test_set)


# In[19]:


print(train_batch_len)
print(val_batch_len)
print(test_batch_len)
print(real_test_batch_len)


# In[20]:


batch_size = 256
train_batch_size = []
val_batch_size = []
test_batch_size = []
real_test_batch_size = []

## Change training batch size
for val in range(batch_size, max(train_batch_len), batch_size):
    train_batch_size.append(val)
for val in train_batch_len:
    train_batch_size.append(val)
    
train_batch_size.sort()

## Change validation batch size
for val in range(batch_size, max(val_batch_len), batch_size):
    val_batch_size.append(val)
for val in val_batch_len:
    val_batch_size.append(val)
    
val_batch_size.sort()

## Change testing batch size
for val in range(batch_size, max(test_batch_len), batch_size):
    test_batch_size.append(val)
for val in test_batch_len:
    test_batch_size.append(val)

test_batch_size.sort()

## Change real testing batch size
for val in range(batch_size, max(real_test_batch_len), batch_size):
    real_test_batch_size.append(val)
for val in real_test_batch_len:
    real_test_batch_size.append(val)

real_test_batch_size.sort()


# # Model

# In[21]:


from keras.layers import Input, Embedding, Dropout, TimeDistributed, Conv1D, Dense, multiply, CuDNNLSTM, GlobalAveragePooling1D
from keras.layers import Concatenate, MaxPooling1D, GlobalMaxPooling1D, Flatten, Bidirectional, LSTM, ThresholdedReLU
from keras.initializers import RandomUniform
from keras.models import Model, load_model
from keras.utils import plot_model,Progbar
from keras.callbacks import EarlyStopping
from keras_multi_head import MultiHeadAttention
from keras import optimizers


# In[22]:


maxwordlength = 15
dropout_p = 0.5
char_embedding_dim = 30


# In[23]:


def get_word_embed(wordEmbeddings):
    ## Input Layer
    words_input = Input(shape = (None,), dtype = np.int32, name = 'words_input')
    ## Embedding Layer
    words = Embedding(input_dim = wordEmbeddings.shape[0],
                      output_dim = wordEmbeddings.shape[1],
                      weights = [wordEmbeddings],
                      trainable = False)(words_input)
    return words_input, words


# In[24]:


def get_char_embed(maxwordlength, char2Idx, char_embedding_dim):
    ## Input Layer
    chars_input = Input(shape = (None, maxwordlength, ), name = 'chars_input')
    ## Embedding Layer
    chars = TimeDistributed(Embedding(input_dim = len(char2Idx),
                  output_dim = char_embedding_dim,
                  embeddings_initializer = RandomUniform(minval=-0.5, maxval=0.5)))(chars_input)
    return chars_input, chars


# In[25]:


def get_model(dropout_p, chars, words):
    ## Convolution layer
    conv = TimeDistributed(Dropout(dropout_p))(chars)
    conv = TimeDistributed(Conv1D(filters = 30,
                                  kernel_size = 3,
                                  padding = 'same',
                                  activation = 'tanh',
                                  strides = 1))(conv)
    conv = TimeDistributed(MaxPooling1D(maxwordlength))(conv)
    ## Flatten
    conv = TimeDistributed(Flatten())(conv)
    chars = TimeDistributed(Dropout(dropout_p))(conv)
    ## Concatenate words embed and char-representation
    output = Concatenate()([words, chars])
    ## Bi-LSTM
    output = Bidirectional(CuDNNLSTM(200, kernel_initializer='random_uniform',
                           bias_initializer='zeros', return_sequences = True))(output)
    ## Dropout
    output = Dropout(dropout_p)(output)
    ## MultiHeadAttention
    attn = MultiHeadAttention(head_num=400, name = 'Multi-head')(output)
    ## LSTM
    output = CuDNNLSTM(200, kernel_initializer='random_uniform',
             bias_initializer='zeros', return_sequences = True)(attn)
    ## Dropout
    output = Dropout(dropout_p)(output)
    ## CRF
    crf = CRF(len(label2idx), sparse_target=True)
    output = crf(output)
    
    return output


# In[26]:


words_input, words = get_word_embed(wordEmbeddings)
chars_input, chars = get_char_embed(maxwordlength, char2Idx, char_embedding_dim)
output = get_model(dropout_p, chars, words)


# In[31]:


## Optimizer
nadam = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model = Model(inputs = [words_input, chars_input], outputs = [output])
model.compile(loss = crf_loss, optimizer = nadam, metrics=[crf_accuracy])
print(model.summary())


# # Training Process

# In[32]:


## Generator
train_generator = iterate_minibatches(train_batch, train_batch_size)
val_generator = iterate_minibatches(val_batch, val_batch_size)
test_generator = iterate_minibatches(real_test_batch, real_test_batch_size)

early_stopping_callback = EarlyStopping(monitor='val_crf_accuracy', mode='max', verbose=1, patience=10)
epochs = 50
model.fit_generator(train_generator, steps_per_epoch=len(train_batch_size), epochs=epochs,
                    callbacks=[early_stopping_callback],
                    validation_data=test_generator, validation_steps=len(real_test_batch_size))


# In[33]:


model.save("../model/ner_new.h5")


# In[34]:


import pickle
pickle.dump(word2Idx,open("../data/pickle_file/word2idx.pkl", "wb"))
pickle.dump(char2Idx, open("../data/pickle_file/char2idx.pkl","wb"))
pickle.dump(label2idx, open("../data/pickle_file/label2idx.pkl","wb"))


# In[35]:


model = load_model("../model/ner_new.h5",
                   custom_objects ={'CRF':CRF,
                                   'crf_loss':crf_loss,
                                   'crf_accuracy':crf_accuracy,
                                   'MultiHeadAttention':MultiHeadAttention})


# In[36]:


def checkaccuracy(predict_label,correct_label):
    count = 0
    for i in zip(predict_label, correct_label):
        if(i[0] != i[1]):
            count += 1
    if(count == 0):
        return 1
    else:
        return 0


# In[37]:


# Performance on dev dataset
predLabels, correctLabels = tag_dataset(test_batch)

# Calculate Performance of model on data
counter = 0
for i in range(len(predLabels)):
    counter += checkaccuracy(predLabels[i], correctLabels[i])
print("Accuracy:", counter/len(predLabels))


# # Fine Tuning using Bayesian Optimization

# In[39]:


get_ipython().system(u'pip install bayesian-optimization --user -q')


# In[49]:


def fit_with(verbose, wordEmbeddings, maxwordlength, char2Idx, char_embedding_dim, train_generator, 
             train_batch_size, test_generator, real_test_batch_size, dropout_p, lr):
    
    # Create the embedding
    words_input, words = get_word_embed(wordEmbeddings)
    chars_input, chars = get_char_embed(maxwordlength, char2Idx, char_embedding_dim)
    
    # Create the model using a specified hyperparameters.
    model = get_model(dropout_p, chars, words)

    # Train the model for a specified number of epochs.
    optimizer = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model = Model(inputs = [words_input, chars_input], outputs = [model])
    model.compile(loss = crf_loss, optimizer = optimizer, metrics=[crf_accuracy])

    # Train the model with the train dataset.
    epochs = 10
    model.fit_generator(train_generator, steps_per_epoch=len(train_batch_size), epochs=epochs)

    # Evaluate the model with the test dataset.
    score = model.evaluate_generator(test_generator, steps=len(real_test_batch_size))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the accuracy.
    return score[1]


# In[50]:


from functools import partial

verbose = 1
fit_with_partial = partial(fit_with, verbose, wordEmbeddings, maxwordlength, char2Idx,
                           char_embedding_dim, train_generator,train_batch_size, test_generator,
                           real_test_batch_size)


# In[52]:


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'dropout_p': (0.35, 0.7), 'lr': (0.001, 0.01)}

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(init_points=3, n_iter=50)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)


# In[ ]:




