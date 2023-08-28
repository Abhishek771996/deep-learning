#!/usr/bin/env python
# coding: utf-8

# In[47]:


import tensorflow as tf  # it is a library used to create neural network.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[48]:


data=pd.read_csv('Churn_Modelling.csv')
data


# In[49]:


data.isnull().sum()


# In[50]:


x=data.iloc[:,3:-1]
x


# In[51]:


y=data['Exited']
y


# In[52]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()                                  
x['Gender']=le.fit_transform(x['Gender']) 
x


# In[53]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encode",OneHotEncoder(drop="first",sparse=False),[1])],remainder="passthrough")
x=ct.fit_transform(x) 


# In[54]:


x


# In[55]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# In[56]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[70]:


# first step of creating neural network
ann=tf.keras.models.Sequential()  



# In[71]:


# working on squential neurals
# input layer                      
ann.add(tf.keras.layers.Input(shape=11))  


# In[72]:


# hidden layer                                 
ann.add(tf.keras.layers.Dense(20,activation='relu'))


# In[74]:


# output layer
ann.add(tf.keras.layers.Dense(1,activation='sigmoid')) 


# In[75]:


# compile
ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])


# In[77]:


# after training this is our neural network model 

model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True)


# In[79]:


# fit
history=ann.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=80,batch_size=32,callbacks=[model_checkpoint]) 


# In[80]:


error= history.history
print(error)  


# In[81]:


plt.plot(range(1,81),error['loss'])
plt.plot(range(1,81),error['val_loss'])
plt.title('val loss and train loss')
plt.xlabel('no of epoches')
plt.ylabel('loss')
plt.legend(['Train loss','val loss'])
plt.show()


# In[45]:






# In[ ]:




