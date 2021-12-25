#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
#%matplotlib inline


# In[2]:


with open('maha5to18.txt','r') as f:
    text = f.read().split('\n\n')


# In[3]:


text[:5]


# In[4]:


text = [sent.replace('\n',' ') for sent in text]


# In[5]:


text[1].split('.')


# In[6]:


bag = [sentence for para in text for sentence in para.split('.') if sentence != '']
bag_size = len(bag)


# In[7]:


bag[:10]


# In[8]:


import random 

sentence_a = []
sentence_b = []
labels = []

for paragraph in text:
    sentences = [sentence  for sentence in paragraph.split('.') if sentence != '']
    num_sentences = len(sentences)
    if num_sentences>1:
        start = random.randint(0, num_sentences-2)
        sentence_a.append(sentences[start])
        if random.random() > 0.5:
            sentence_b.append(sentences[start+1])
            labels.append(0)
        else:
            sentence_b.append(bag[random.randint(0,bag_size-1)])
            labels.append(1)


# In[9]:


sentence_a[:5],labels[:5]


# In[10]:


sentence_b[:5],labels[:5]


# In[11]:


from transformers import BertTokenizer, BertForPreTraining


# In[12]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[13]:


model = BertForPreTraining.from_pretrained('bert-base-uncased')


# In[14]:


inputs = tokenizer(sentence_a,sentence_b,return_tensors = 'pt',max_length = 512,truncation = True,padding = 'max_length')


# In[15]:


inputs.keys()


# In[16]:


inputs['next_sentence_label'] = torch.LongTensor([labels]).T
inputs['next_sentence_label'][:10]


# In[17]:


inputs['labels'] = inputs.input_ids.detach().clone()


# In[18]:


inputs.keys()


# In[19]:


rand  = torch.rand(inputs.input_ids.shape)


# In[20]:


mask_array = (rand < 0.15)*(inputs.input_ids != 101)*(inputs.input_ids != 102)*(inputs.input_ids != 0)


# In[21]:


for i in  range(inputs.input_ids.shape[0]):
    selection = torch.flatten(mask_array[i].nonzero()).tolist()
    inputs.input_ids[i,selection]  = 103


# In[22]:


class MahabharatDataset(torch.utils.data.Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


# In[23]:


dataset = MahabharatDataset(inputs)


# In[24]:


loader = torch.utils.data.DataLoader(dataset,batch_size = 8,shuffle=True)


# In[25]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[26]:


torch.cuda.empty_cache()


# In[27]:


model.to(device)


# In[28]:


model.train()


# In[29]:


from transformers import AdamW


# In[30]:


optimizer = AdamW(model.parameters(),lr = 1e-5)


# In[31]:


state = torch.load('my_model.pt')


# In[35]:


checkpoint = torch.load('my_model.pt')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
#loss = checkpoint['loss']


# In[ ]:


from tqdm import tqdm

epochs= 1
for epoch in range(epochs):
    loop = tqdm(loader,leave = True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask= batch['attention_mask'].to(device)
        token_type_ids= batch['token_type_ids'].to(device)
        next_sentence_label= batch['next_sentence_label'].to(device)
        labels= batch['labels'].to(device)
        outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,next_sentence_label=next_sentence_label,labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch: {epoch}')
        loop.set_postfix(loss = loss.item())
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(checkpoint, 'my_model.pt',_use_new_zipfile_serialization=False)


# In[65]:


torch.save(model, 'my_saved_model.pt')


# In[67]:


model = torch.load('my_saved_model.pt')


# In[ ]:




