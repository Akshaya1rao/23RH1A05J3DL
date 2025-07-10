import tensorflow as tf
from tensorflow.keras.preprocessing.image image ImageDataGenerator
from tensorflow.keras.models import Seqeuntial
from tensorflow.keras.layers import Conv2D,MaxPolling2D,Flatten,dense,Dropout
import os
IMG_size=128
BATCH_SIZE=30
train_ds_path="/path/train"
test_ds_path="/path/test"
#creating pexcels for images
train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
train_data=train_datagen.flow_from_dirctory(train_ds_path)
test_data=test_datagen.flow_from_dirctory(test_ds_path)
#create model
model=Seqeuntial([Conv2D(32,(3,3)),activation='relu',input_size=(IMG_size,IMG_size,3),MaxPolling2D(),
Conv2D(64,(3,3)),activation='relu',MaxPolling2D(),Flatten(),Dense(128,activation='relu'),Dropout(0,3),Dense(train_data.num_classes,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_data,validation_data=test_data,epochs=10)
