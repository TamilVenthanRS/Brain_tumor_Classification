
# <p align="center">BRAIN TUMOR CLASSIFICATION

## *Introduction* :
The project on brain tumor classification utilizing Convolutional Neural Networks (CNN) aims to develop a sophisticated system for accurate and automated classification of brain tumors. CNNs are deep learning models well-suited for image analysis tasks. By training a CNN on a large dataset of brain tumor images, the system can learn to identify complex patterns and features indicative of different tumor types. The CNN architecture allows for hierarchical feature extraction, enabling the system to capture intricate details within the images. The project's goal is to leverage the power of CNNs to create a reliable and efficient tool that can aid medical professionals in diagnosing brain tumors with improved accuracy and efficiency.
  
## _Purpose of the Project_ :
The purpose of the project on brain tumor classification using Convolutional Neural Networks (CNN) is to develop an advanced and automated system that can accurately classify brain tumors based on medical imaging data. By leveraging the power of CNNs, the project aims to enhance the accuracy and efficiency of brain tumor diagnosis, enabling medical professionals to make informed decisions regarding treatment planning and patient management. The system's ability to extract intricate features and patterns from brain tumor images can assist in distinguishing between different tumor types, such as benign and malignant tumors. Ultimately, the project seeks to improve patient outcomes by providing a reliable and accessible tool for brain tumor classification.
  
## _Objective_ :
The objective of the project on brain tumor classification using Convolutional Neural Networks (CNN) is threefold. Firstly, it aims to develop a robust and accurate classification system that can differentiate between different types of brain tumors based on medical imaging data. Secondly, the project aims to optimize the CNN architecture and training process to achieve high performance and efficiency in tumor classification. Lastly, the objective is to create a user-friendly interface that allows medical professionals to easily input brain tumor images, obtain classification results, and gain valuable insights to aid in diagnosis and treatment planning. The overall goal is to provide a reliable and accessible tool for improved brain tumor classification and patient care.
  
## _Abstract_ :
The project on brain tumor classification using Convolutional Neural Networks (CNN) focuses on developing an automated system for accurately classifying brain tumors based on medical imaging data. By leveraging the power of CNNs, the project aims to enhance the accuracy and efficiency of brain tumor diagnosis. The CNN architecture enables hierarchical feature extraction, allowing the system to identify intricate patterns and characteristics specific to different tumor types. Through extensive training on a large dataset of labeled brain tumor images, the system can learn to differentiate between benign and malignant tumors. The project's objective is to provide medical professionals with a reliable tool that aids in accurate brain tumor classification, ultimately improving patient outcomes and treatment planning.
  

## _Methodology_ :
+ The necessary libraries are imported, including Keras, numpy, pandas, OpenCV, and matplotlib.
+ The dataset can be extracted. The extracted dataset contains training and testing images categorized into different tumor classes.
+ The code initializes empty lists to store training data and labels, The code initializes empty lists to store training data and labels.
+ Similar to the training set, testing set  can be prepared by iterating through each tumor class in the testing directory. 
+ Using the technique of thresholding and various morphological operations like erosion and dilation, the tumor area can be detected.
+ The CNN model architecture is modelled and compiled. The model is trained on training data and gets tested by testing data. Atlast, the model will accurately predict the type of brain tumor.



  
  
## _Project FlowChart :
  
 ![Collect the scan report image (3)](https://github.com/TamilVenthanRS/Brain_tumor_Classification/assets/75235477/5ef22676-3e9a-460d-b297-c03fbce3a4e9)

## Algorithm :

Certainly! Here's an algorithmic representation of the given code:

1. Import the required libraries and modules: `keras`, `os`, `PIL`, `numpy`, `sklearn`, `matplotlib.pyplot`, `pandas`, and `cv2`.

2. Define and initialize the `OneHotEncoder` object `enc` with four categories: 0, 1, 2, and 3.

3. Define a function `names()` that maps numbers to corresponding class names. The class names are assumed to be defined elsewhere in the code.

4. Prepare the training data and labels:
   - Set the image dimensions to (150, 150).
   - Specify the training data path.
   - Iterate through each directory in the training path.
   - Inside each directory, iterate through the files and extract the image paths.
   - Open each image, resize it to the specified dimensions, convert it to a numpy array, and add it to the `trainData` list.
   - Transform the index value to a one-hot encoded label using the `enc` object and add it to the `trainLabel` list.
   - Print the corresponding class name and directory name.
   - Increment the index value.

5. Convert the `trainData` and `trainLabel` lists to numpy arrays.

6. Prepare the test data and labels:
   - Set the image dimensions to (150, 150).
   - Specify the testing data path.
   - Iterate through each directory in the testing path.
   - Inside each directory, iterate through the files and extract the image paths.
   - Open each image, resize it to the specified dimensions, convert it to a numpy array, and add it to the `testData` list.
   - Transform the index value to a one-hot encoded label using the `enc` object and add it to the `testLabel` list.
   - Print the corresponding class name and directory name.
   - Increment the index value.

7. Convert the `testData` and `testLabel` lists to numpy arrays.

8. Define the CNN model using `Sequential` from Keras:
   - Add a convolutional layer with 32 filters, a kernel size of (3, 3), and an input shape of (150, 150, 3).
   - Apply ReLU activation function.
   - Add a max-pooling layer with a pool size of (2, 2).
   - Add another convolutional layer with 32 filters and a kernel size of (3, 3).
   - Apply ReLU activation function.
   - Add another max-pooling layer with a pool size of (2, 2).
   - Flatten the output of the previous layer.
   - Add a fully connected layer with 32 units.
   - Apply ReLU activation function.
   - Apply dropout with a rate of 0.25.
   - Add a fully connected layer with 4 units for the output classes.
   - Apply softmax activation function.

9. Compile the model using categorical cross-entropy loss and the Adam optimizer. Print the model summary.

10. Train the model on the training data and labels:
    - Use a batch size of 32 and train for 14 epochs.
    - Provide the testing data and labels as the validation data.

11. Perform Binary thresholding:
    - Read an image using `cv2.imread`.
    - Convert the image to RGB color space.
    - Apply a binary threshold to the image.
    - Perform erosion, dilation, and morphological opening on the binary image.
    - Display the original image, binary image, and processed image using `matplotlib.pyplot`.


  
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
In conclusion, the project on brain tumor classification using Convolutional Neural Networks (CNN) has successfully developed an automated system that accurately classifies brain tumors based on medical imaging data. By leveraging the power of CNNs, the system improves the accuracy and efficiency of brain tumor diagnosis. This advancement provides medical professionals with a reliable tool for enhanced brain tumor classification and improved patient care.
  
## _Results_ :
The project on brain tumor classification using Convolutional Neural Networks (CNN) achieved promising results. The CNN-based system demonstrated high accuracy in distinguishing between different types of brain tumors based on medical imaging data. The system's ability to extract intricate features and patterns from the images contributed to its success. These results highlight the potential of CNNs in improving the accuracy and efficiency of brain tumor classification for enhanced patient care and treatment planning.

## _References_ :

+ [Brain tumor detection based on Naïve Bayes Classification](https://ieeexplore.ieee.org/abstract/document/8802562)
by hein tun zaw.

+ [Brain Tumor Types Classification using K-means Clustering and ANN Approach](https://ieeexplore.ieee.org/document/9331115) by Angona Biswas and Md. Saiful Islam.

+ [Brain Tumor Classification Using Support Vector Machines](https://ieeexplore.ieee.org/document/8318767) by Amruta Hebli.

+ [Brain Tumor Classification Using Support Vector  Machines  with Kernel Principal Component Analysis](https://www.researchgate.net/publication/337077979_Brain_Tumor_Classification_Using_Principal_Component_Analysis_and_Kernel_Support_Vector_Machine) by Richard Torres-Molina.
