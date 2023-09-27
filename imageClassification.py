#Modules for use in this lab.
#Note you cannot use additional modules, 
#unless they are standard libraries (i.e., "os").
import tarfile
import requests
import os
import numpy as np
import pickle
import keras

#=========================================
#=========================================
#LAB QUESTION 8
#=========================================
#=========================================
#In this question, you are going to write
#and fit the best network you can to solve a problem.
#You will then save your model, and submit
#"Q8.h5" alongside your submission.py.
#Your grade for this question will be based on
#the accuracy of your saved model.

#To write this model, you must use keras, as
#per the below example.  We will be using
#the UC Merced Land Use database.
#You will find the images in the folder "mercerImages"
#In this assignment.

#Once you fit your model and upload it, it will
#be tested against a completely distinct
#set of images from another dataset you do not
#have access to, but are of the same dimensions and classes.  
#The accuracy of your model in predicting
#this second dataset is what will determine
#your score on this question.

#Notes:
#1) It is totally up to you to subdivide your images
#into test/train sets.  In this example, I only use a training
#dataset.  This is obviously wrong, don't do this!

#2) You don't actually need to submit your submissionNet
#Code - your grade on this question is just based on the Q8.h5 file
#you submit.

#3) You *must* include metrics=['categorical_accuracy'] in your
#modile compilation (i.e., see below).

#4) If you achieve 50% accuracy on the independent test set,
#you will receive a 100% on this question. 

#5) The maximum model size is 100 MiB (a gradescope limitation).
#So, the *.h5 file you save with your model must be no more than 100MiB.

def exampleNet():
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=64,
                              kernel_size=(4,4),
                              activation="tanh",
                              input_shape=(256,256,3)))
    m.add(keras.layers.GlobalAveragePooling2D())
    m.add(keras.layers.Dense(units=21))
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=.001),
                                            loss='categorical_hinge',
                                            metrics=['categorical_accuracy'])
    return(m)

dataGenerator = tensorflow.keras.preprocessing.image.ImageDataGenerator(validation_split=0.3)

test = dataGenerator.flow_from_directory("./mercerImages", class_mode='categorical', batch_size=32, subset="validation")
train = dataGenerator.flow_from_directory("./mercerImages", class_mode='categorical', batch_size=32, subset="training")
    
model = exampleNet()
model.fit(train, epochs=5, validation_data = test)
model.save("Q8.h5")