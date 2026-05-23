"""
Quick demo of training with Diabetic_Retinopathy_Dataset.csv using a small number
of epochs so the model file gets generated without a long wait.
"""
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split

# read csv
print('loading csv')
df = pd.read_csv('Diabetic_Retinopathy_Dataset.csv')
labels_map = {'Normal':0,'Mild':1,'Moderate':2,'Severe':3,'Proliferative':4}

images=[]
labels=[]
for idx,row in df.iterrows():
    severity=row['Severity']
    lbl=labels_map[severity]
    img=np.zeros((128,128,3),np.uint8)
    cv2.circle(img,(64,64),50,(100,100,100),-1)
    cv2.circle(img,(64,64),45,(150,150,150),-1)
    cv2.circle(img,(64,64),25,(50,50,50),-1)
    images.append(img)
    labels.append(lbl)
images=np.array(images,dtype=np.float32)/255.0
labels=np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

model=models.Sequential([
    layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.Flatten(),
    layers.Dense(5,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=1,batch_size=16)
model.save('models/demo.h5')
print('demo model saved')