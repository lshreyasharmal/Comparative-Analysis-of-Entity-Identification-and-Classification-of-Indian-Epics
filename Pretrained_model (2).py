#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline
plt.style.use("ggplot")


# In[2]:


annotations = pd.read_csv("chap4_mahabharath.csv", index_col=['id'])


# In[3]:


annotations.head()


# In[4]:


words = list(annotations['token'].values)
words.append('PADword')
n_words = len(set(words))
n_words, len(words)


# In[5]:


tags = list(set(annotations["tag"].values))
n_tags = len(tags)
print(n_tags)
tags


# In[6]:


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence#").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None


# In[7]:


getter = SentenceGetter(annotations)
sent = getter.get_next()
print(sent)


# In[8]:


sentences = getter.sentences
print(len(sentences))


# In[9]:


sentences[0][0][1]


# In[10]:


largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))


# In[11]:


plt.hist([len(s) for s in sentences], bins = 50)
plt.show()


# In[12]:


max_len = 50
X = [[str(w[0]) for w in s] for s in sentences]


# In[13]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
living_entity_tags = ['ANIMAL','PERSON','GROUP','TITLE']
non_living_entity_tags = ['BOOK','PLACE','WEAPON','SPECIAL_OBJECT','PLANT','CONCEPT','WATER']

#for extraction of entities
tags2index = {}
for tag in tags:
    if tag not in living_entity_tags and tag not in non_living_entity_tags:
        tags2index[tag] = 0
    elif tag in living_entity_tags:
        tags2index[tag] = 1
    else:
        tags2index[tag] = 2


# In[14]:


Y = [[tags2index[w[1]] for w in s] for s in sentences]
Y = pad_sequences(maxlen=max_len, sequences=Y, value=0, padding='post', truncating='post')


# In[15]:


Y[0]


# In[16]:


X_sent=[]
for i in range(len(X)):
    X_sent.append(X[i])


# In[17]:


X_join=[]
for i in range(len(X_sent)):
    X_join.append(" ".join(X_sent[i]))


# In[18]:


len(X_join)


# In[19]:


import tensorflow as tf


# In[20]:


import keras
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import Model, Input
#from keras.models import load_model
from tensorflow.keras.layers import add, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,concatenate
import seqeval
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# In[21]:


X_tr, X_te, y_tr, y_te = train_test_split(X_join, Y, test_size=0.1, random_state=2021)
batch_size = 32


# In[22]:


from transformers import BertTokenizer
bert = 'farhanjafri/my-model'

tokenizer = BertTokenizer.from_pretrained(bert, do_lower_case=True, add_special_tokens=True,
                                                max_length=max_len, paddind=True)


# In[23]:


def tokenize(sentences, tokenizer):
    input_ids = []
    input_masks = []
    for sent in sentences:
        inputs = tokenizer.encode_plus(sent, 
                                    add_special_tokens=True,
                                    max_length=50,
                                    pad_to_max_length = True, 
                                    return_attention_mask=True,
                                    return_token_type_ids=True,
                                    truncation=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
     #input_segments.append(inputs['token_type_ids'])        
       
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')



# In[24]:


len(X_tr), len(X_te), batch_size,len(X_tr)/batch_size, len(X_te)/batch_size 


# In[25]:


X_tr, X_val = X_tr[:865*batch_size], X_tr[-100*batch_size:]
y_tr, y_val = y_tr[:865*batch_size], y_tr[-100*batch_size:]
#y_tr = y_tr.reshape(y_tr.shape[0], 1)
#y_val = y_val.reshape(y_val.shape[0], 1)


# In[26]:


X_tr_in,X_tr_mask = tokenize(X_tr,tokenizer)
X_val_in,X_val_mask = tokenize(X_val,tokenizer)
X_te_in,X_te_mask = tokenize(X_te,tokenizer)
#y_tr_in,y_tr_mask,_ = np.array(tokenize(y_tr,tokenizer))
#y_val_in,y_val_mask,_ = np.array(tokenize(y_val,tokenizer))


# In[27]:


X_tr_in


# In[28]:


np.array(y_tr).shape


# In[29]:


from transformers import TFBertForPreTraining
transformer_model =TFBertForPreTraining.from_pretrained('farhanjafri/my-model')


# In[30]:


#from transformers import AutoModel
#transformer_model =AutoModel.from_pretrained('farhanjafri/my-model')

input_ids_in=tf.keras.layers.Input(shape=(max_len,),name='input_token',dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(max_len,),name='masked_token',dtype='int32')
#input_segments_in = tf.keras.layers.Input(shape=(max_len,),name='segment_token',dtype='int32')

embedding_layer = transformer_model([input_ids_in,input_masks_in])[0]
#cls_token = embedding_layer[:,:]
#dense = Dense(786,activation = 'relu')(cls_token)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                      recurrent_dropout=0.2, dropout=0.2))(embedding_layer)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                          recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  
x = Dense(2056,activation='relu')(x)
#out = Dense(1024, activation="relu")(cls_token)
out = Dense(3, activation="softmax")(x)
model = Model([input_ids_in,input_masks_in], out)

for layer in model.layers[:3]:
     layer.trainable = False

model.summary()


# In[31]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[32]:




# In[33]:


from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import EarlyStopping

#checkpoints = ModelCheckpoint('checkpoints.h5', monitor='loss', save_best_only=True, verbose=1, mode='min')
#early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='max', baseline=None, restore_best_weights=False)
callbacks = [PlotLossesKeras()]


# In[34]:


X_tr_in = tf.convert_to_tensor(X_tr_in)
X_tr_mask = tf.convert_to_tensor(X_tr_mask)
X_val_in = tf.convert_to_tensor(X_val_in)
X_val_mask = tf.convert_to_tensor(X_val_mask)
y_tr = tf.convert_to_tensor(y_tr)
y_val = tf.convert_to_tensor(y_val)


# In[35]:


history = model.fit([X_tr_in,X_tr_mask],y_tr,
                    validation_data=([X_val_in,X_val_mask],y_val),
                    batch_size=batch_size, 
                    epochs=5,
                    callbacks=[callbacks], 
                    verbose=1)


# In[36]:


X_te = X_te[:107*batch_size]
y_te = y_te[:107*batch_size]
X_te_in,X_te_mask = np.array(tokenize(X_te,tokenizer))


# In[37]:


preds = model.predict([X_te_in,X_te_mask], verbose=1, batch_size=batch_size)


# In[38]:


# in case of classification
# idx2tag = {i: w for w, i in tags2index.items()}

# in case of extraction of entities
idx2tag = {}
for k,v in tags2index.items():
    if v == 0:
        idx2tag[v] = 'O'
    elif v == 1:
        idx2tag[v] = "LIVING"
    else:
        idx2tag[v] = "NON-LIVING"
        
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out

def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out


# In[39]:


pred_labels = pred2label(preds)


# In[40]:


np.array(pred_labels).shape


# In[41]:


y_te.shape


# In[42]:


test_labels = test2label(y_te)


# In[43]:


print(np.array(test_labels).shape)


# In[44]:


print(classification_report(pred_labels, test_labels))


# In[45]:


print(f1_score(pred_labels, test_labels,average='micro'))


# In[ ]:


print("Job complete")

