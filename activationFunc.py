
import tarfile
import requests
import os
import numpy as np
import pickle
import keras

#FUNCTION NAME: forwardTanh
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

def forwardTanh(x):
    out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    cache = x
    return(out, cache)

#FUNCTION NAME: forwardSigmoid
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

def forwardSigmoid(x):
    out = 1 / (1 + np.exp(-x))
    cache = x
    return(out, cache)

#FUNCTION NAME: forwardLeakyRelu
#Parameters: x (arbitrary dimensions, numpy)
#Outputs: out, cache, where cache is the incoming x, 
#and out is the output of the activation layer.

#Example input:
#testInput = np.array([10,20,30,-400])

def forwardLeakyRelu(x):
    out = np.maximum(0.01 * x, x)
    cache = x
    return(out, cache)

#FUNCTION NAME: backwardLeakyRelu
#Parameters: upstreamGradient and cache (same dimensions; np arrays)
#In this case, the cache holds the "x" value from the forward pass.
#Outputs: array / matrix / tensor of gradients to pass further upstream.

#Example input:
#cache = np.array([10,20,30,-400])
#upstreamGradient = np.array([3.1, .04, -13.3, 104.2])

def backwardLeakyRelu(upstreamGradient, cache):
    x = cache
    dx = np.array(upstreamGradient, copy = True)
    dx[x <= 0] = 0.01
    return(dx)