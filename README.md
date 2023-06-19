
# <p align="center">BRAIN TUMOR CLASSIFICATION

## *Introduction* :
  
## _Purpose of the Project_ :
  
## _Objective_ :
  
## _Abstract_ :
  

## _Methodology_ :
  
  
## _Project FlowChart :
  
 ![Collect the scan report image (3)](https://github.com/TamilVenthanRS/Brain_tumor_Classification/assets/75235477/5ef22676-3e9a-460d-b297-c03fbce3a4e9)

## Algorithm :
  
## _Program_ :
  
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3]]) 
def names(number):
    if(number == 0):
        return classes[0]
    elif(number == 1):
        return classes[1]
    elif(number == 2):
        return classes[2]
    elif(number == 3):
        return classes[3]
trainData = []
trainLabel = []
dim = (150, 150)
trainPath = "/content/Brain-Tumor-Classification-DataSet-master/Training"
index = 0
for dir in os.listdir(trainPath):
    filePaths = []
    subDir = os.path.join(trainPath, dir)
    for file in os.listdir(subDir):
        imgFullPath = os.path.join(subDir, file)
        filePaths.append(imgFullPath)
        img = Image.open(imgFullPath)
        x = img.resize(dim)
        x = np.array(x)
        trainData.append(np.array(x))
        trainLabel.append(enc.transform([[index]]).toarray())
    print(names(index))
    print(str(dir))
    index += 1
    
trainData = np.array(trainData)
trainLabel = np.array(trainLabel).reshape(2870, 4)
print(trainData.shape)
print(trainLabel.shape)
testData = []
testLabel = []
dim = (150, 150)
testPath = "/content/Brain-Tumor-Classification-DataSet-master/Testing"
index = 0
for dir in os.listdir(testPath):
    filePaths = []
    subDir = os.path.join(testPath, dir)
    for file in os.listdir(subDir):
        imgFullPath = os.path.join(subDir, file)
        filePaths.append(imgFullPath)
        img = Image.open(imgFullPath)
        x = img.resize(dim)
        x = np.array(x)
        testData.append(np.array(x))
        testLabel.append(enc.transform([[index]]).toarray())
    print(names(index))
    print(str(dir))
    index += 1
testData = np.array(testData)
testLabel = np.array(testLabel).reshape(394, 4)
print(testData.shape)
print(testLabel.shape)
  
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu'))           
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='adam')
print(model.summary())
  
history = model.fit(trainData, trainLabel,batch_size = 32, epochs = 14, verbose=1,validation_data=(testData, testLabel))
  


original_image = cv2.imread("/content/Brain-Tumor-Classification-DataSet-master/Testing/pituitary_tumor/image(15).jpg")
im=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
ret,th=cv2.threshold(im, 170 ,255,cv2.THRESH_BINARY)


kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN , kernel)
figure_size=7
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,3,1),plt.imshow(im)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(th)
plt.title('Binary image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(opening)
plt.title('Eroded+Dilated img'), plt.xticks([]), plt.yticks([])
plt.show()

img = Image.open('/content/Brain-Tumor-Classification-DataSet-master/Testing/pituitary_tumor/image(15).jpg')
dim = (150, 150)
x = np.array(img.resize(dim))
x = x.reshape(1,150,150,3)
answ = new_model.predict_on_batch(x)
classification = np.where(answ == np.amax(answ))[1][0]
imshow(img)
print(str(answ[0][classification]*100) + '% Confidence This Is ' + names(classification))
```
  
## Output:
  ![Untitled design](https://github.com/TamilVenthanRS/Brain_tumor_Classification/assets/75235477/8e650171-9c7e-4b65-99ab-f8c990a59841)

  
  
  ![2023-06-19 (1)](https://github.com/TamilVenthanRS/Brain_tumor_Classification/assets/75235477/71b9d806-21e2-4f15-b01b-fa3e677324ed)

## _Conclusion_ :
  
## _Results_ :
