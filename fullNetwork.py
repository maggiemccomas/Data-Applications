
import tarfile
import requests
import os
import numpy as np
import pickle
import keras


#FUNCTION NAME: submissionNet
#Parameters: None.
#Returns a fully defined keras model 

#After you submit, the model is fit on a node that can run for no more than
#approximately 20 minutes. The calls that are performed are:
#model = submission.submissionNet()
#X_train is shape (40000, 32, 32, 3)
#y_train has been converted to a categorical, and is shape (40000,10)
#model.fit(x=X_train, y=y_train,
#          batch_size=512,
#          epochs=5)
#model.evaluate(X_test, y_test)

#***The primary challenge for this question is to build a model which is both
#small enough to train on limited infrastructure, but powerful enough to 
#get a somewhat-reasonable accuracy.  

#You *cannot* use predefined networks to accomplish this task; rather, you 
#must define your own here.  To get a 100%, you must hit a baseline
#categorical accuracy of 35% within the 5 epoch limit.  You must achieve
#at least 20% for any credit at all.

#IMPORTANT: Your model compile must include metrics=['categorical_accuracy']

def submissionNet():
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters=512,
                              kernel_size=(2,2),
                              activation="relu",
                              input_shape=(32,32,3)))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.MaxPooling2D())  
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(units=10))
    
    
    m.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.001), metrics = ['categorical_accuracy'], loss = 'categorical_hinge')


    return(m)




