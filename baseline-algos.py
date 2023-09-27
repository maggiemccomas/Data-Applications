
import tarfile
import requests
import os
import numpy as np
import pickle


#FUNCTION NAME: dataDownload 
#FUNCTION DESCRIPTION: Function to download CIFAR10 data.
#FUNCTION OUTPUT: Path, as a string, of the location CIFAR10 was extracted to.
#FUNCTION NOTES: 
#1) Download the file https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#2) Extract the file to a path (i.e., "./images")
#3) Return the path that the files were extracted to

def dataDownload(url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 outPath = "./images/"):
    r = requests.get(url)
    open("cifar-10-python.tar.gz", 'wb').write(r.content)
    tarfile.open("cifar-10-python.tar.gz").extractall(path=outPath)

    return(outPath)


#FUNCTION NAME: dataSplit
#FUNCTION DESCRIPTION: Function to ingest and split the CIFAR data
#FUNCTION OUTPUT: Path, as a string, to the pickle containing training and test data.
#FUNCTION NOTES: 
#1) Load each pickled data_batch in the CIFAR dataset.
#2) Reshape and transpose the training data (i.e., 10k observations, 3 bands, 32x32 dimensions)
#3) Repeat this for your test data
#4) Save and pickle all of your results.

#The final format of the variable should look like:
#labData = {}
#labData["X_train"] = X_train
#labData["y_train"] = y_train
#labData["X_test"] = X_test
#labData["y_test"] = y_test
#pickle.dump(labData, f)

def dataSplit(basePath = "./images/",
              picklePath = "./testTrainLab1.pickle"):
    
    cifar10_dir = basePath + "cifar-10-batches-py"
    
    xs = []
    ys = []
    
    for b in range(1,6):
        d = os.path.join(cifar10_dir, 'data_batch_%d' % (b, ))
        
        with open(d, 'rb') as f:
            datadict = pickle.load(f, encoding = 'latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y = np.array(Y)
            
        xs.append(X)
        ys.append(y)
        
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    with open(os.path.join(cifar10_dir, "test_batch"), 'rb') as f:
        datadict = pickle.load(f, encoding = 'latin1')
        X = datadict['data']
        Y = datadict['labels']
        X_test = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y_test = np.array(Y)
    
    with open(picklePath, 'wb') as f:
        labData = {}
        labData["X_train"] = X_train
        labData["y_train"] = y_train
        labData["X_test"] = X_test
        labData["y_test"] = y_test
        pickle.dump(labData, f)
    
    return(picklePath)

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):        
        Ypred = np.zeros(len(X), dtype=np.dtype(self.ytr.dtype))

        for i in range(0, len(X)):
            l1Distances = np.sum(np.abs(self.Xtr - X[i]), axis=1)
            minimumDistance = np.argmin(l1Distances)
            Ypred[i] = self.ytr[minimumDistance]
        
        return Ypred

#CLASS NAME: knnClassifier
#FUNCTION DESCRIPTION: Function to classify an image based on a knn Classifer with a L2 norm.
#FUNCTION OUTPUT: Class prediction for a list of images, between 0 and 9.
#FUNCTION NOTES: 

class knnClassifier:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X, k):
        Ypred = np.zeros(len(X), dtype=np.dtype(self.ytr.dtype))

        for i in range(0, len(X)):
            l2Distances = np.sqrt(np.sum((self.Xtr - X[i])**2, axis = 1))
            
            minimumIndices = np.argsort(l2Distances)
            kClosest = minimumIndices[:k]
            predClass, counts = np.unique(self.ytr[kClosest], return_counts = True)
            Ypred[i] = predClass[counts.argmax()]
        
        return Ypred

#FUNCTION NAME: crossFoldValidation
#FUNCTION DESCRIPTION: Function to conduct a cross fold validation on a given classifier
#                      for a single set of hyperparameters.
#REQUIRED FUNCTION PARAMETERS:
#                      modelToValidate - the model to validate, i.e. knnClassifier()
#                      X_train - the full (unsplit) X training dataset
#                      y_train - the full (unsplit) y training dataset
#                      folds - an integer representing the number of folds
#FUNCTION OUTPUT: List of accuracy values - one per fold.
#FUNCTION NOTES: 
#The value of k = 5 should be *hard coded*

def crossFoldValidation(modelToValidate,
                        X_train,
                        y_train,
                        folds=5):
    
    k = 5
    
    X_folds = np.array_split(X_train, folds)
    y_folds = np.array_split(y_train, folds)
    
    accuracies = []

    for i in range(0,folds): 
        classifier = knnClassifier()
        classifier.train(np.concatenate(np.delete(X_folds, [i], axis = 0)), np.concatenate(np.delete(y_folds, [i], axis = 0)))
        predictions = classifier.predict(X_folds[i], k = k)
        correctCases = sum(predictions == y_folds[i])
        accuracy = correctCases / len(y_folds[i])
        accuracies.append(accuracy)

    return(accuracies)  

#Function Name: svmClassifier
#FUNCTION DESCRIPTION: Function to solve for the multiclass SVM data loss (with regularization).
#REQUIRED FUNCTION PARAMETERS:
#                      X - the full (unsplit) X training dataset
#                      y - the full (unsplit) y training dataset
#                      W - a vector of weights (i.e., W = np.random.randn(3072, 10) * 0.0001)
#                      e - epsilon term for SVM loss
#                      l - Lambda for regularization loss
#FUNCTION OUTPUT: Library with 'dataLoss', 'regLoss', 'totalLoss', and loss matrix loss_ij.

def svmClassifier(X, y, W, e, l):
    scores = X.dot(W)
    countTrainSamples = scores.shape[0]
    trueClassScores = scores[np.arange(scores.shape[0]), y]
    trueClassMatrix = np.matrix(trueClassScores).T 
    loss_ij = np.maximum(0, (scores - trueClassMatrix) + e) 
    loss_ij[np.arange(countTrainSamples), y] = 0

    dataLoss = np.sum(loss_ij) / countTrainSamples
    regLoss = np.sum(W * W)
    totalLoss = dataLoss + (l * regLoss)
    
    return({'dataLoss':dataLoss, 'regLoss':regLoss, 'totalLoss':totalLoss, 'loss_ij':loss_ij})


#Function Name: svmOptimizer
#FUNCTION DESCRIPTION: Function to optimize parameters W in the function svmClassifier
#FUNCTION OUTPUT: Best identified set of parameters W for input svm model (3072 * 10).

def svmOptimizer(X, y, model = svmClassifier):
    W = np.random.randn(3072, 10) * .0001
    e = 1
    l = 1
    currentIteration = 1
    maxIterations = 1000
    learningRate = .0000001

    while currentIteration < maxIterations:
        
        iterationResult = model(X, y, W, e, l)
        gradient = X.T.dot(iterationResult['loss_ij']) / X.shape[0]
        W += -learningRate * gradient
            
        currentIteration = currentIteration + 1

    return(W)


#=========================================
#Function tests
#=========================================
if __name__ == '__main__':
  print(dataDownload())
  print(dataSplit())

  with open("testTrainLab1.pickle", "rb") as f:
    labData = pickle.load(f)
    X_train = np.reshape(labData["X_train"], (labData["X_train"].shape[0], -1))
    X_test = np.reshape(labData["X_test"], (labData["X_test"].shape[0], -1))
    y_train = labData["y_train"]
    y_test = labData["y_test"]

    y_test = y_test[:250]
    X_test = X_test[:250]
    y_train = y_train[:250]
    X_train = X_train[:250]

  print(crossFoldValidation(folds=5,
                        modelToValidate = knnClassifier(),
                        X_train = X_train,
                        y_train = y_train))

  W = np.random.randn(3072, 10) * 0.0001
  print(svmClassifier(X_train, y_train, W, e=1, l=1))

  print(svmOptimizer(X=X_train, y = y_train))
