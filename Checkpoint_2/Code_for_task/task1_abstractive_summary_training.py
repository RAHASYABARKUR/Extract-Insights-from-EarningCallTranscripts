#GEID 1011139547
#Rahasya Barkur
from __future__ import division
import numpy as np
import pickle

import pdb

import gensim
from gensim.models import Word2Vec
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional
import numpy as np
from keras.callbacks import ModelCheckpoint
import numpy as np
import pdb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

##############
## Training ##
##############

with open("Data//train_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("Data//train_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

binary_output = []
correct_ex = []
#sentence=sentence[0:20000]
#compressed_sentence=compressed_sentence[0:20000]
print(len(sentence))

print(len(compressed_sentence))
for i in range(len(sentence)):
    bin_vec = np.zeros((len(sentence[i].split()),1))
    words_in_sentence = sentence[i].split()
    words_in_comp_sent = compressed_sentence[i].split()
    #print(sentence[i])
    #print(compressed_sentence[i])
    #print(words_in_sentence)
    #print(words_in_comp_sent)
    correct_ex.append(1)
    for j in range(len(words_in_comp_sent)):
        try:
            bin_vec[words_in_sentence.index(words_in_comp_sent[j])] = 1

        except:
            try:
                bin_vec[words_in_sentence.index(words_in_comp_sent[j].lower())] = 1
            except:
                print(words_in_comp_sent[j])
                correct_ex[-1] = 0
                
    
    binary_output.append(bin_vec)
    #print(sum(correct_ex),i)
    #print(i)    
    #print(sentence[i])
    #print(bin_vec)
    #print(compressed_sentence[i])    
#print(i)
print(len(binary_output))

with open("Data//train_binary_output.txt","wb") as fp: #Binary output for the sentences.
    pickle.dump(binary_output,fp)

np.save('Data//train_correct_examples.npy',correct_ex) #Index of Sentences which are correct.


#############
## Testing ##
#############

with open("Data//test_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("Data//test_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

binary_output = []
correct_ex = []

for i in range(len(sentence)):
    bin_vec = np.zeros((len(sentence[i].split()),1))
    words_in_sentence = sentence[i].split()
    words_in_comp_sent = compressed_sentence[i].split()
    #print(sentence[i])
    #print(compressed_sentence[i])
    #print(words_in_sentence)
    #print(words_in_comp_sent)
    correct_ex.append(1)
    for j in range(len(words_in_comp_sent)):
        try:
            bin_vec[words_in_sentence.index(words_in_comp_sent[j])] = 1
            
        except:
            try:
                bin_vec[words_in_sentence.index(words_in_comp_sent[j].lower())] = 1
            except:
                #print("testproblem")
                print(words_in_comp_sent[j])
                

    binary_output.append(bin_vec)
    #print(sum(correct_ex),i)    
    #print(i)
    #print(sentence[i])
    #print(bin_vec)
    #print(compressed_sentence[i])    

with open("Data//test_binary_output.txt","wb") as fp: #Binary output for the sentences.
    pickle.dump(binary_output,fp)

np.save('Data//test_correct_examples.npy',correct_ex) #Index of Sentences which are correct.
######################################################
## Remove the examples which have problems in them. ##
######################################################


##############
## Training ##
##############

with open("Data//train_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("Data//train_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

with open("Data//train_binary_output.txt","rb") as fp:
    bin_output = pickle.load(fp)

correct_ex = np.load('Data//train_correct_examples.npy')

###############################
## Filter too long sentences ##
###############################
sen_len = []
for i in range(len(bin_output)):
    sen_len.append(bin_output[i].shape[0])
sen_len = np.array(sen_len)

print("Mean Length of Sentences: ",np.mean(sen_len))
print("Std Length of Sentences: ",np.std(sen_len))

## We will not take examples with length greater than max_sen_len ##
#min_sen_len = int(np.mean(sen_len) - 1*np.std(sen_len))
#max_sen_len = int(np.mean(sen_len) + 1*np.std(sen_len))

min_sen_len = 15
max_sen_len = 40

sen_len_check = []
for i in range(len(bin_output)):
    if bin_output[i].shape[0] > max_sen_len or bin_output[i].shape[0] < min_sen_len:
        sen_len_check.append(0)
    else:
        sen_len_check.append(1)

refined_sen = []
refined_comp_sen = []
refined_bin_output = []
refined_len = []

for i in range(len(correct_ex)):
    if correct_ex[i] == 1 and sen_len_check[i] == 1:
        refined_sen.append(sentence[i])
        refined_comp_sen.append(compressed_sentence[i])
        refined_bin_output.append(bin_output[i])
        refined_len.append(bin_output[i].shape[0])

print("Max Length :", np.max(np.array(refined_len)))
#pdb.set_trace()
print(len(refined_sen),len(refined_comp_sen),len(refined_bin_output))

print("Total no of sentences within this range :", len(refined_sen))

with open("Data//refined_train_sentence.txt","wb") as fp:
    pickle.dump(refined_sen,fp)

with open("Data//refined_train_compressed_sentence.txt","wb") as fp:
    pickle.dump(refined_comp_sen,fp)

with open("Data//refined_train_binary_output.txt","wb") as fp:
    pickle.dump(refined_bin_output,fp)



#############
## Testing ##
#############


with open("Data//test_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("Data//test_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

with open("Data//test_binary_output.txt","rb") as fp:
    bin_output = pickle.load(fp)

correct_ex = np.load('Data//test_correct_examples.npy')

###############################
## Filter too long sentences ##
###############################
sen_len = []
for i in range(len(bin_output)):
    sen_len.append(bin_output[i].shape[0])
sen_len = np.array(sen_len)

print("Mean Length of Sentences: ",np.mean(sen_len))
print("Std Length of Sentences: ",np.std(sen_len))

## We will not take examples with length greater than max_sen_len ##
#min_sen_len = int(np.mean(sen_len) - 1*np.std(sen_len))
#max_sen_len = int(np.mean(sen_len) + 1*np.std(sen_len))

min_sen_len = 15
max_sen_len = 40

sen_len_check = []
for i in range(len(bin_output)):
    if bin_output[i].shape[0] > max_sen_len:
        sen_len_check.append(0)
    else:
        sen_len_check.append(1)
        


refined_sen = []
refined_comp_sen = []
refined_bin_output = []
refined_len = []

for i in range(len(correct_ex)):
    if correct_ex[i] == 1 and sen_len_check[i] == 1:
        refined_sen.append(sentence[i])
        refined_comp_sen.append(compressed_sentence[i])
        refined_bin_output.append(bin_output[i])
        refined_len.append(bin_output[i].shape[0])
#pdb.set_trace()

print("Total no of sentences in this range:", len(refined_sen))
print(len(refined_sen),len(refined_comp_sen),len(refined_bin_output))
print("Max Length: ",np.max(np.array(refined_len)))
with open("Data//refined_test_sentence.txt","wb") as fp:
    pickle.dump(refined_sen,fp)

with open("Data//refined_test_compressed_sentence.txt","wb") as fp:
    pickle.dump(refined_comp_sen,fp)

with open("Data//refined_test_binary_output.txt","wb") as fp:
    pickle.dump(refined_bin_output,fp)


##############
## Training ##
##############

with open("Data//train_sentence.txt","rb") as fp:
    train_sentence = pickle.load(fp)

with open("Data//test_sentence.txt","rb") as fp:
    test_sentence = pickle.load(fp)


sentences = []

for i in range(len(train_sentence)):
    sentences_split = train_sentence[i].split()
    sentences.append(sentences_split)

for i in range(len(test_sentence)):
    sentence_split = test_sentence[i].split()
    sentences.append(sentence_split)

print("Total length of the corpus we have is :",len(sentences))

#model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin')
#print("Model loading done")

model = Word2Vec(sentences, min_count=1,size=300)
model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin.gz',lockf=1.0,binary=True)
model.train(sentences,total_examples=len(sentences),epochs=20)
model.wv.save_word2vec_format('Data/google_finetuned.bin',binary=True)

#model = Word2Vec(sentences, min_count=1,size=256)
#model.wv.save_word2vec_format('../../my_word2vec.bin',binary=True)

#pdb.set_trace()


#For sentences in 10-15
#no_train = 15000
#no_test = 500

#For sentences in 20-25
#no_train = 20000
#no_test = 1000
embedding_size = 300


#####################
## Gensim Word2Vec ##
#####################

from gensim.models import Word2Vec

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
model1 = gensim.models.KeyedVectors.load_word2vec_format('Data/google_finetuned.bin', binary=True)

print("Loading Word2Vec done ...")

dirlist = ["train","test"]
predir = "Data//refined_"

for direc in dirlist:

    with open(predir+direc+"_sentence.txt","rb") as fp:
        sentence = pickle.load(fp)

    with open(predir+direc+"_compressed_sentence.txt","rb") as fp:
        compressed_sentence = pickle.load(fp)

    with open(predir+direc+"_binary_output.txt","rb") as fp:
        bin_output = pickle.load(fp)

    print(len(sentence),len(compressed_sentence),len(bin_output))
    #################################################
    ## To get the max sentence length in training. ##
    #################################################

    
    """max_len = len(bin_output[0])
    for i in range(len(bin_output)):
        if bin_output[i].shape[0] > max_len:
            max_len = bin_output[i].shape[0]
    print(max_len)
    """
    max_len = 40

    ###################################
    ## Generate train decoder output ##
    ###################################
    decoder_output = []
    for i in range(len(bin_output)):
        #print("Decoder Output: ",i)
        if i < len(sentence):
            temp_bin = np.zeros((max_len,1))
            temp_bin[:bin_output[i].shape[0],0] = bin_output[i][:,0]
            decoder_output.append(temp_bin)
        else:
            break

    decoder_output = np.array(decoder_output)
    np.save(predir+direc+"_decoder_output.npy",decoder_output)
    print("decoder_output: ",decoder_output.shape)
    del decoder_output

    print('\n',direc," decoder output ......",'\n')


    ##################################
    ## Generate train encoder input ##
    ##################################
    
    if len(sentence) != len(bin_output):
        print("Kuch toh pain hai")

    encoder_input = np.zeros((len(sentence),max_len,embedding_size))
    for i in range(len(sentence)):
        #print("Encoder Input: ",i)
        if i < len(sentence):
            sentence_em = []
            words_in_sentences = sentence[i].split()
            words_in_sentences = words_in_sentences[::-1]
            for j in range(max_len):
                word_em = np.zeros((1,embedding_size))
                if j < len(words_in_sentences):
                    try:
                        word_em = model[words_in_sentences[j]]
                        word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                        #pdb.set_trace()
                    except:
                        yo=1
                        word_em = model1[words_in_sentences[j]]
                        word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                encoder_input[i,j,:] = word_em
        else:
            break
    
    #encoder_input = np.array(encoder_input)
    np.save(predir+direc+"_encoder_input.npy",encoder_input)
    print("encoder_input: ",encoder_input.shape)
    del encoder_input

    print('\n',direc," encoder input ......",'\n')

    #pdb.set_trace()

    ##################################
    ## Generate train decoder input ##
    ##################################

    decoder_input = np.zeros((len(sentence),max_len,embedding_size))
    for i in range(len(sentence)):
        #print("Decoder Input: ",i)
        if i < len(sentence):
            sentence_em = []
            words_in_sentences = sentence[i].split()
            for j in range(max_len):
                word_em = np.zeros((1,embedding_size))
                if j < len(words_in_sentences):
                    try:
                        word_em = model[words_in_sentences[j]]
                        word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                    except:
                        yo=1
                        word_em = model1[words_in_sentences[j]]
                        word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                decoder_input[i,j,:] = word_em
        else:
          break
    np.save(predir+direc+"_decoder_input.npy",decoder_input)
    print("decoder_input: ",decoder_input.shape)
    del decoder_input

    print("\n",direc," decoder input ......","\n")





epochs = 30
embedding_dim = 300
sentence_len = 40
batch_size = 8

###################
## Weights paths ##
###################

#expName = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run = "30epochs"

best_weights = 'WeightsD/pure_best_weights_exp_'+str(run)+'.h5'
last_weights = 'WeightsD/pure_last_weights_exp_'+str(run)+'.h5'
#initial_weights = '../Weights/best_weights_2.h5'

###############
## Load data ##
###############

train_encoder_input = np.load('Data//refined_test_encoder_input.npy')
train_decoder_input = np.load('Data//refined_test_decoder_input.npy')
train_decoder_output = np.load('Data//refined_test_decoder_output.npy')
#pdb.set_trace()

######################
## Compressive LSTM ##
######################

drop = 0.2

inputs = Input(shape=(sentence_len,embedding_dim))
lstm_1 = LSTM(64, return_sequences=True,dropout=drop,recurrent_dropout=drop,
kernel_regularizer=regularizers.l2(0.00))(inputs)

lstm_2 = Bidirectional(LSTM(32, return_sequences=True,dropout=drop,
recurrent_dropout=drop,kernel_regularizer=regularizers.l2(0.00)))(lstm_1)

lstm_3 = Bidirectional(LSTM(32, return_sequences=True,dropout=drop,
recurrent_dropout=drop,kernel_regularizer=regularizers.l2(0.00)))(lstm_2)

dense_1 = TimeDistributed(Dense(1,activation='sigmoid'))(lstm_3)

model = Model(inputs,dense_1)
#sgd = SGD(lr=0.001, momentum=0.01, decay=0.0, nesterov=False)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath= best_weights, monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)

#class_weight = {0:0.3,1:0.7} 


def train():

    # Run training
    model.summary()
    model.fit(train_decoder_input, train_decoder_output,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          #class_weight=class_weight,
          callbacks=[checkpointer])

    #del train_encoder_input,train_decoder_input,train_decoder_output
    model.save_model(last_weights)



train()








