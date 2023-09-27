
import tarfile
import requests
import os
import numpy as np
import pickle
import keras


#FUNCTION NAME: convolutionalForward
#Parameters: X - shape is (N, Height, Width, Channels)
#            W - shape is (F, filterSize, filterSize, imageChannels)
#            B - shape is (F,)
#Stride should be hard coded to 2.
#No pooling.

#Outputs: One return (out), which is a shape (N, F) output of activation surfaces
#that have been maxpooled (just like the example implementation).  No cache.

#Example input:
#exampleInput = X_train[0:10]
#exampleInput.shape outputs (10, 32, 32, 3)

def convolutionalForward(X, W, B, stride=2):
    (N, Height, Width, Channels) = X.shape
    (F, filterSize, filterSize, imageChannels) = W.shape
    
    convH = int((Height - filterSize) / stride) + 1
    convW = int((Width - filterSize) / stride) + 1
    
    activationSurface = np.zeros((N, F, convH, convW, imageChannels))
    out = np.zeros((N,F))
    
    for i in range(N):
        x = X[i]
        for f in range(F):
            for h in range(0, Height, stride):
                for w in range(Width):
                    for c in range(Channels):
                        y_upper = h * stride
                        y_lower = y_upper + filterSize
                        x_left = w * stride
                        x_right = x_left + filterSize
                    
                        window = x[y_upper: y_lower, x_left: x_right, :]
                    
                        if ((window.shape[0] == filterSize) and window.shape[1] == filterSize):
                            s = np.multiply(window, W[f])
                            activationSurface[i, f, h, w, c] = np.sum(s)
                            activationSurface[i, f, h, w, c] = activationSurface[i, f, h, w, c] + B[f]
                        
                            out[i, f] = np.max(activationSurface[i, f, h, w, c])

    return(out)