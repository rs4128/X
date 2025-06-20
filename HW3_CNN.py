#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Convolutional Neural Networks
# 
# Due Wednesday 11/24 at 11:59 pm EST

# Download the dataset `cats-notcats` from github (given as a part of the assignment). This dataset has images of cats and images that are not cats (in separate folders). The task is to train a convolutional neural network (CNN) to build a classifier that can classify a new image as either `cat` or `not cat`

# In[128]:


import PIL
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding,Flatten, LSTM, Dense, Input, MaxPooling2D, Conv2D, Concatenate, Activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Load the dataset and create three stratified splits - train/validation/test in the ratio of 70/10/20. 

# In[113]:


#code here
base_dir_cats = '/Users/rahulsubramaniam/assignment-3-rs4128/data/cats-notcats/cats/'
base_dir_notcats = '/Users/rahulsubramaniam/assignment-3-rs4128/data/cats-notcats/notcats/'
base_dir_train = '/Users/rahulsubramaniam/assignment-3-rs4128/data/cats-notcats/train/'
base_dir_val = '/Users/rahulsubramaniam/assignment-3-rs4128/data/cats-notcats/val/'
base_dir_test = '/Users/rahulsubramaniam/assignment-3-rs4128/data/cats-notcats/test/'

cat_image_names = os.listdir(base_dir_cats)
cat_image_names = filter(lambda cat_image_name: cat_image_name.endswith('.jpg'),cat_image_names )
notcat_image_names = os.listdir(base_dir_notcats)
notcat_image_names = filter(lambda notcat_image_name: notcat_image_name.endswith('.jpg'),notcat_image_names )
cat_image_paths = [base_dir_cats + s for s in cat_image_names]
not_cat_image_paths = [base_dir_notcats + s for s in notcat_image_names]
os.mkdir(base_dir_train)
os.mkdir(base_dir_val)
os.mkdir(base_dir_test)


# In[114]:


def get_image(image_name,base_dir):
    image = PIL.Image.open(os.path.join(base_dir, image_name))
    return np.asarray(image.resize((40,40))) / 255.0

def get_image_with_path(image_path):
    image = PIL.Image.open(image_path)
    return np.asarray(image.resize((40,40))) / 255.0

def img_generator(img_list):
  
  for img_val in img_list:
    img = get_image(img_val)
    yield np.array([img])
    


# In[115]:


cats_df = pd.DataFrame({'image_path':cat_image_paths,'is_cat':[1]*len(cat_image_paths)})
notcats_df = pd.DataFrame({'image_path': not_cat_image_paths,'is_cat':[0]*len(not_cat_image_paths)})
df = pd.concat([cats_df,notcats_df],axis=0)
X_dev,X_test,y_dev,y_test = train_test_split(df['image_path'],df['is_cat'],test_size=0.2,stratify=df['is_cat'])
X_train,X_val,y_train,y_val = train_test_split(X_dev,y_dev,test_size=0.125,stratify=y_dev)


# In[116]:


y_train


# In[117]:


cat_dir = 'isCat/'
notcat_dir = 'isNotCat/'
for i in range(len(list(X_train))):
    image_path = list(X_train)[i]
    splits = image_path.split('/')
    if list(y_train)[i] ==1:
        shutil.copy(image_path,base_dir_train+cat_dir+splits[len(splits)-1])
    else:
        shutil.copy(image_path,base_dir_train+notcat_dir+splits[len(splits)-1])
for i in range(len(list(X_val))):
    image_path = list(X_val)[i]
    splits = image_path.split('/')
    if list(y_val)[i] ==1:
        shutil.copy(image_path,base_dir_val+cat_dir+splits[len(splits)-1])
    else:
        shutil.copy(image_path,base_dir_val+notcat_dir+splits[len(splits)-1])
    
for i in range(len(list(X_test))):
    image_path = list(X_test)[i]
    splits = image_path.split('/')
    if list(y_test)[i] ==1:
        shutil.copy(image_path,base_dir_test+cat_dir+splits[len(splits)-1])
    else:
        shutil.copy(image_path,base_dir_test+notcat_dir+splits[len(splits)-1])
    


# In[122]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(base_dir_train,target_size=(40,40),batch_size=20,class_mode='binary')
val_generator = test_datagen.flow_from_directory(base_dir_val,target_size=(40,40),batch_size=20,class_mode='binary')


# 2. Create a CNN that has the following hidden layers:
# 
#     a. 2D convolution layer with a 3x3 kernel size, has 128 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     b. Max pooling layer of 2x2
#     
#     c. Dense layer with 128 dimensions and ReLU as the activation layer

# In[135]:


#code here
model = Sequential()
model.add(Conv2D(128,(3,3),strides=(1, 1),padding='same',input_shape=(40,40,3),name='conv_layer_1'))
model.add(MaxPooling2D((2,2),name='maxpool_1'))
model.add(Flatten())
model.add(Dense(128,activation='relu',name='dense_1'))
model.add(Dense(1,activation='sigmoid',name='output'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[136]:


model.summary()


# 3. Train the classifier for 20 epochs with 100 steps per epoch. Also use the validation data during training the estimator.

# In[137]:


#code here
history = model.fit_generator(train_generator,steps_per_epoch = 100, validation_data=val_generator,validation_steps=14, verbose = True, epochs = 20)


# 4. Plot the accuracy and the loss over epochs for train & validation sets

# In[146]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy for validation and train')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.show()


# In[147]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for validation and train')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train','val'])
plt.show()


# In[148]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Loss for validation and train')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train','val'])
plt.show()


# 5. Add the following layers to (2) before the dense layer:
# 
#     a. 2D convolution layer with a 3x3 kernel size, has 64 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     b. Max pooling layer of 2x2
#     
#     c. 2D convolution layer with a 3x3 kernel size, has 32 filters, stride of 1 and padded to yield the same size as input, followed by a ReLU activation layer
#     
#     d. Max pooling layer of 2x2
#     
#     e. Dense layer with 256 dimensions and ReLU as the activation layer

# In[151]:


#code here
model = Sequential()
model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same',input_shape=(40,40,3),name='conv_layer_1',activation='relu'))
model.add(MaxPooling2D((2,2),name='maxpool_1'))
model.add(Conv2D(32,(3,3),strides=(1, 1),padding='same',name='conv_layer_2',activation='relu'))
model.add(MaxPooling2D((2,2),name='maxpool_2'))
model.add(Flatten())
model.add(Dense(256,activation='relu',name='dense_1'))
model.add(Dense(1,activation='sigmoid',name='output'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# 6. Train the classifier again for 20 epochs with 100 steps per epoch. Also use the validation data during training the estimator.

# In[152]:


#code here
history = model.fit_generator(train_generator,steps_per_epoch = 100, validation_data=val_generator,validation_steps=14, verbose = True, epochs = 20)


# 7. Plot the accuracy and the loss over epochs for train & validation sets

# In[153]:


#code here
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for validation and train')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train','val'])
plt.show()


# In[154]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy for validation and train')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.show()


# In[68]:


# train_image_arrays = np.empty(shape=[299,299,3])
# for image_path in X_train:
#     image_array = get_image_with_path(image_path)
#     #train_image_arrays = np.append(train_image_arrays,image_array,axis = 0)
#     train_image_arrays =  np.vstack((train_image_arrays,image_array))
# train_image_arrays


# In[ ]:




