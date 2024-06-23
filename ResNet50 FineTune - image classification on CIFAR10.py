'''

use resnet50 PRETRAINED MODEL on CIFAR10 dataset


'''

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets,models,Sequential,layers
from tensorflow.python.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
import numpy as np 


(x_train,y_train), (x_test,y_test) = datasets.cifar10.load_data() 

x_train.shape
y_train.shape
x_test.shape
y_test.shape

# now normalize our data
x_train = x_train/255 
x_test = x_test/255
                                






                                '''
                                # HERE WE USE TRANSFER LEARNING TECHNIQUE
                                # WE USE RESNET50 PRETRAINED MODEL
                                # HERE WE DONE FINE TUNNING
                                
                                '''
                                
                                
resnet_model = Sequential()

pretrained_model = keras.applications.ResNet50(
                    include_top= False,        # False = bcoz this model is trained on different dimension
                    input_shape =(32,32,3),    # our image shape
                    classes=9,                 # 9 classes present in my dataset
                    pooling= 'avg',            
                    weights="imagenet" )

for layer in pretrained_model.layers:   # False :basically means that whatever Resnet 50 model learnt, do not learn this weight again 
    layer.trainable = False             # keep this weight as it is.  this save more time complexity in training process.

resnet_model.add(pretrained_model)                     
resnet_model.add(Flatten())                            # Flatten layer
resnet_model.add(Dense(512, activation = 'relu'))      # Dense Hidden layer
resnet_model.add(Dense(10, activation = 'softmax'))    # Dense Output layer




resnet_model.summary()

resnet_model.compile(loss = 'sparse_categorical_crossentropy',
                     optimizer = 'adam', 
                     metrics= ['accuracy']) 




resnet_model.fit(x_train, y_train, epochs = 10)   # training accuracy 

# resnet_model.fit(x_train, y_train, epochs = 10, batch_size= 32)   # train the model with batch size.


'''
# EarlyStopping:- when the accuracy of model is not increasing and loss is not decreasing that time automatically training of the model will be stopped.
from tensorflow.keras.callbacks import EarlyStopping  

# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Monitor validation loss and stop after 5 epochs with no improvement

# Train the model with Early Stopping
resnet_model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])

'''


















resnet_model.evaluate(x_test,y_test)              # testing accuracy













#  ACTUAL

x_test[1]                # 1st image in pixel format
plt.imshow(x_test[1])    # dispaly the image
y_test[1]                # label of 1st image (which class belongs to this image)
# multiclass_classification
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 
classes[8]  



# PREDICT (lets check how our model is predict test data) 

y_pred = resnet_model.predict(x_test) 

y_pred[1]                          # here we get probability of each class
y_predwith_maxprob = [np.argmax(element) for element in y_pred]  # convert into class number
y_predwith_maxprob[1] 


# NOW CHECK ACCURACY OF OF MULTICLASS CLASSIFICATION

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_predwith_maxprob)
report = classification_report(y_test,y_predwith_maxprob) 


