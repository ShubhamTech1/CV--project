
'''
build CNN from scratch
TASK :- IMAGE CLASSIFICATION, MODEL = CNN(base model), DATASET = CIFAR10.
 
'''




import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np 

# load the data
(x_train,y_train), (x_test,y_test) = datasets.cifar10.load_data()   

#checking the shape of trainning and testing data
x_train.shape  # 50k images pixel values in the form of array RGB.
y_train.shape  # classes of 50k images (which classses belong to this image)  
x_test.shape   # 10k images pixel values in the form of array RGB.
y_test.shape   # classes of 10k images (0-9) 


# multiclass_classification
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 
classes[9] 


#check which class belongs to 1st image. 
plt.imshow(x_train[1]) 


# here i check first  10 sample (which is our predicted value Y)
y_train[:10]  # this is 2D array convert into 1D and access classes of each image at any index
y_train = y_train.reshape(-1,) 
y_train[:10]    # here we convert into 1D array, now access the class element


def plot_sample(X, y, index):
    plt.imshow(X[index])                # give me a image at index position.  input_X
    plt.xlabel(classes[y[index]])       # give me a (label,Y,output,predicted result) on that index.  OUTPUT_Y
    
plot_sample(x_train,y_train, 1000)    #  i got it 


# y_train[1]




'''
NOW WE NORMALIZE THE IMAGE (both x_train, x_test = bcoz here is pixels of different different images present then pass to cnn model)

ANN is less effective for image dataset, its effecive on tabular data 
reduce the pixel values in between 0-1

'''


# NORMALIZE BOTH IMAGES TRAIN AND TEST
x_train[0]            #here we get all pixel values of 0th image
x_train = x_train/255
x_test = x_test/255











'''

NOW WE BUILD OUR CNN MODEL

'''
from tensorflow.keras.initializers import glorot_uniform, he_normal

cnn = models.Sequential([
#   layers.Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (32,32,3),kernel_initializer=glorot_uniform), # here we can use also he_normal.
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (32,32,3),kernel_initializer=he_normal), # he_normal weight initialization technique is very handy with relu activation function.

    layers.BatchNormalization(), # Batch normalization = IT helps to neural networks learn faster. sometimes gradients are huge and small(bcoz of in hidden layer use relu activation function and output layer use sigmoid and other activation function).batch normalization helps to gradients are smoothly reach the global minima.
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'), 
    layers.MaxPooling2D((2,2)),
    
    layers.Dropout(0.2),  # reduce overfitting, dropout layer like L1,L2 Regularization to reduce overfitting.
                          # randomly drop certain proportion of neurons and this neurons not used in forward pass and backward pass.
                          
    
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),                      
    layers.Dense(10, activation = 'softmax')
    ])


cnn.compile(loss= 'sparse_categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])
          
'''
WITH DEFINE LEARNING RATE(but this is wrong scenario, we can use learning rate Schedulers)

from tensorflow.keras import optimizers 
# Define the optimizer with the desired learning rate
optimizer = optimizers.Adam(learning_rate = 0.1)

# Compile the model with the optimizer and metrics
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
'''

#------------------------------------------------------------------------------

'''
LEARNING RATE SCHEDULER PROCESS:-

from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Define the optimizer (initial learning rate)
optimizer = Adam(learning_rate=0.01)  # Adjust initial learning rate as needed

# Define the learning rate scheduler (ReduceLROnPlateau)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            factor=0.5,
                                            min_lr=0.0001)

cnn.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# Train the model with the learning rate scheduler
cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

'''




cnn.fit(x_train, y_train, epochs = 10)    # training accuracy


cnn.evaluate(x_test, y_test)              # testing accuracy





# VISUALLY CHECK

plot_sample(x_test,y_test,1)  # i want to check what is my 1st testing image, before we reshape our y_test array
y_test[:7]  #check 1st 7 testing images and their class Y
y_test = y_test.reshape(-1,)
y_test[:10]  # done reshaping






# now predict this image , and check how our model is performed
plot_sample(x_test, y_test, 0)  #identify which is my 1st testing image -ACTUAL IMAGE

# NOW CHECK HOW MODEL THIS PREDICT
y_pred = cnn.predict(x_test)
y_pred[:1]  # here we get predicted value in the form of probability of all 10 classes. Because softmax  gives the probability of each classes
y_test[:1]  # actual value

y_predwith_maxprob = [np.argmax(element) for element in y_pred]  

''' 
now we check y_test(actual),y_predwith_maxprob(predicted) 
'''


y_test[:5]  # actual 1st five images

y_predwith_maxprob[:5] # after predicting 1st five images


plot_sample(x_test, y_test, 200)  # actual value
y_predwith_maxprob[200]           # predicted value
classes[3]                        # predicted value 

# in above case we get error because --> actual = dog, predict = cat
# because as a model it is very difficult to identify  whether it is cat or dog and our testing accuracy is also low.



# classification Report
from sklearn.metrics import classification_report,confusion_matrix
report = classification_report(y_test,y_predwith_maxprob)
cm = confusion_matrix(y_test, y_predwith_maxprob)





#----------------------------------------------------------------------------------------------------
# KERAS TUNER:- HYPERPARAMETER TUNNING IN D.L

# pip install optuna
# pip install keras-tuner

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner import Tuner
from kerastuner import RandomSearch 
# from KerasTuner import Optuna
# from KerasTuner import Hyperband
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner import RandomSearch

# Define the model building function (can be refactored for clarity)
def build_model(hp):
  # ... (other hyperparameter definitions)
  kernel_size_options = [(3, 3), (5, 5)]
  kernel_size = hp.Choice('kernel_size', min_value=4, max_value=8)
  
  # ... (rest of the model definition)
  filters = 32
  
  model = Sequential([
      Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(32, 32, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Dropout(0.2),
      Flatten(),
      Dense(64, activation='relu'),
      Dropout(0.3),
      Dense(10, activation='softmax')
     
  ])

  model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
  return model

# Define the search tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',  # We want to maximize validation accuracy
    max_trials=10,  # Adjust the number of trials as needed
    executions_per_trial=1
)

# Start the hyperparameter search
tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))  # Adjust epochs as needed

# Get the best model
best_model = tuner.get_best_models()[0]

# Evaluate the best model
best_model.evaluate(x_test, y_test)


































