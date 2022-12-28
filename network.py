import numpy as np
import csv 
import matplotlib.pyplot as plt
import os
from PIL import Image
from numpy import asarray
import copy
import math
#mwuahhahahahha it works! :)))

#l2 regularization
#dropout
#mini-batch
def main():
    showcost = 200
    #hyperparameters loading data:
    maxDataTest = 100
    imageSize = 120
    maxData = 600

    #hyperparameters network
    mini_batch_size = 200
    keepProb = 1
    lambd = 0.05 #between 0 - 0.1 the bigger lambda the bigger penalty on complicated weights
    layerDims = [imageSize * imageSize * 3, 400, 300, 200, 10,50,30,1]
    iterations = 2000
    learningRate = 0.001



    Xtrain, Ytrain = loadImages("train", imageSize, maxData)
    Xtrain, Ytrain = hussle(Xtrain, Ytrain) #for final training error
    mini_batches = random_mini_batches(Xtrain,Ytrain, mini_batch_size)
    Xtest, Ytest = loadImages("test", imageSize, maxDataTest)
    Xtest, Ytest = hussle(Xtest, Ytest)

    parameters = trainmodel(layerDims, mini_batches, iterations, learningRate, showcost, lambd, Xtest, Ytest, keepProb)
    trainingPrediction = makePrediction(Xtrain, parameters) #no dropout for test time
    testPrediction = makePrediction(Xtest, parameters) #no dropout for test time
    print("trainingAccuracy", accuracy(trainingPrediction, Ytrain))
    print("testAccuracy", accuracy(testPrediction, Ytest))

def random_mini_batches(X,Y, mini_batch_size):
    m = X.shape[1]
    mini_batches = []
    shuffled_X, shuffled_Y = hussle(X,Y)
    inc = mini_batch_size
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*inc:(k+1) * inc]
        mini_batch_Y = shuffled_Y[:, k*inc:(k+1) * inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    #handling non-complete mini_batch at the end:
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def loadImages(folder, imageSize, maxData):
    catData = np.zeros((imageSize * imageSize * 3,1))
    counter = 0
    Y = np.full((1, maxData), int(1))
    for images in os.listdir(folder):
        #add 200 cat pictures
        if counter >= maxData:
            break
        counter +=1
        imageVector = processImage(images, folder, imageSize)
        catData = np.append(catData, imageVector, axis = 1)
    catData = catData[:,1:]
    catData = np.append(catData, Y, axis = 0)
    
    Y = np.full((1,maxData), int(0))
    dogData = np.zeros((imageSize * imageSize * 3,1))
    counter = 0
    for images in reversed(os.listdir(folder)):
        #adds 200 dogg pictures
        if counter >= maxData:
            break
        counter +=1
        imageVector = processImage(images, folder, imageSize)
        dogData = np.append(dogData, imageVector, axis = 1)
    dogData = dogData[:,1:]
    dogData = np.append(dogData, Y, axis = 0)
    
    Data = np.append(dogData, catData, axis = 1) #glue them together horizontally ofcourse

    #now randomize data:
    transposedData = Data.T
    np.random.shuffle(transposedData)
    Data = transposedData.T
    labels = np.array([Data[imageSize * imageSize * 3]])
    examples = np.delete(Data, imageSize * imageSize * 3, 0)
    examples = examples / 255
    return examples, labels

def processImage(images, folder, imageSize):
    #takes image returns one dimensional (vertical) image vector
        image = Image.open(os.path.join(folder, images)).resize((imageSize, imageSize)) 
        imageVector = asarray(image)
        imageVector = imageVector.reshape(imageVector.shape[0] * imageVector.shape[1] * imageVector.shape[2], 1)
        return imageVector

def hussle(X,Y):
    size = X.shape[0]
    Data = np.append(X, Y, axis = 0)
    transposedData = Data.T
    np.random.shuffle(transposedData)
    Data = transposedData.T
    labels = np.array([Data[size]])
    examples = np.delete(Data, size, 0)
    return examples, labels

def accuracy(prediction, Y):
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    difference = Y-prediction
    incorrect = np. count_nonzero(difference)
    false = incorrect / Y.shape[1] * 100
    accuracy = 100 - false
    return accuracy



def trainmodel(layerDims, mini_batches, iterations, learningRate, showcost, lambd, Xtest, Ytest, keepProb = 1):
    initialShowCost = showcost
    costs = []
    iterationlist = []
    #parameter initialization (xavier)
    initializeX, initializeY = mini_batches[0]
    parameters, Vdw, Vdb, Sdw, Sdb = parameterInitialization(initializeX, layerDims)
    counter = 0
    for i in range(iterations):
        print(i)
        for mini_batch in mini_batches:
            X,Y = mini_batch
            counter +=1
            iterationlist.append(counter)
            AL, forward_cache, D = L_model_forward(X, parameters, keepProb)
        
            #compute cost (maybe try out L2)
            thiscost = cost(AL, Y, parameters, lambd )
            costs.append(thiscost)
            #backward propagation (with gradient checking)
            grads = L_model_backward(AL, Y, forward_cache, lambd, parameters, keepProb, D)
            #gradCheck:
      
            #updating weights
            parameters = updateWeights(grads, parameters, layerDims, learningRate)
        if i == showcost:
            plt.title("cost function")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.plot(iterationlist, costs, color ="red")
            plt.show()
            showcost += initialShowCost
            trainingPrediction = makePrediction(X, parameters) #no dropout for test time
            testPrediction = makePrediction(Xtest, parameters) #no dropout for test time
            print("trainingAccuracy", accuracy(trainingPrediction, Y))
            print("testAccuracy", accuracy(testPrediction, Ytest))
    return parameters

def cost(A, Y, parameters, lambd):
    m = Y.shape[1]
    losses = -(np.multiply(np.log(A), Y) + np.multiply((1-Y), np.log(1-A)))
    cost = 1/m * np.sum(losses)
    frobeniusNorm = 0
    for l in range(1, int(len(parameters)/2) + 1):
        frobeniusNorm += np.sum(parameters["W" + str(l)] * parameters["W" + str(l)])
    L2_regularizationCost = lambd / (2*m) * frobeniusNorm
    cost += L2_regularizationCost
    return cost #checked


def makePrediction(X, parameters, keepProb = 1):
    AL, _, D = L_model_forward(X, parameters, keepProb)
    return AL

def updateWeights(grads, parameters, layerDims, learningRate):
    m = len(parameters) / 2
    for l in range(1, len(layerDims)):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learningRate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learningRate * grads["db" + str(l)]
    return parameters


def parameterInitialization(X, layerDims):
    numPreviousNeurons = X.shape[0]
    parameters = {}
    for l in range(1, len(layerDims)):
        parameters["W" + str(l)] = np.random.randn(layerDims[l], layerDims[l-1]) * np.sqrt(1/layerDims[l-1])
        parameters["b" + str(l)] = np.zeros((layerDims[l], 1))
        numPreviousNeurons = layerDims[l]
    Vdw = 0
    Vdb = 0
    Sdw = 0
    Sdb = 0
    return parameters, Vdw, Vdb, Sdw, Sdb
    
def L_model_forward(X, parameters, keepProb):
    caches = []
    A = X
    #no dropout in input layer
    L = int(len(parameters) / 2)
    D = []
    for l in range(1, L):
        A_prev = A
        A, cache = linearActivationForward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        #store d one cache ahead for backward prop
        A,d = dropout(A, keepProb)
        D.append(d)
        #linear activation forward returns the next A for the next layer but also returns a cache containing
        #linear (W, A_PREV, b) and activation (Z, "activation" => "relu") cache
        caches.append(cache)
    AL, cache = linearActivationForward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    #no dropout in output layer
    caches.append(cache)
    return AL, caches, D

def dropout(A, keepProb):
    d = np.random.rand(A.shape[0], A.shape[1]) < keepProb
    A = np.multiply(A,d)
    A /= keepProb
    return A, d

def L_model_backward(AL, Y, caches, lambd, parameters, keepProb, D):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    dAl = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    currentCache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAl, currentCache, "sigmoid")
    currentD = D[-1]
    dA_prev_temp = np.multiply(dA_prev_temp, currentD)/keepProb
    #grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    for l in reversed(range(L-1)):
        dA = dA_prev_temp
        currentCache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, currentCache, "relu")
        if l != 0:
            #no dropout for input layer when l == 0
            currentD = D[l - 1]
            dA_prev_temp = np.multiply(dA_prev_temp, currentD) / keepProb
        #grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp + lambd/m * parameters["W" + str(l + 1)]
        grads["db" + str(l+1)] = db_temp
        #the l-1 makes sense, you give in the A of the l+1 in linearactivationbackward,
        #this function will return the previous a which is therefore l,
        #l-1 because you calculate current a with the next weights (see page 5 notes)
        #same logic goes for why L-1 when calculating dA[l-1]
    return grads

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    W = linear_cache["W"]
    b = linear_cache["b"]
    A_prev = linear_cache["A_prev"]
    Z = activation_cache["Z"]
    m = Z.shape[1]

    
    if activation == "relu":
        A = relu(Z)
        dZ = reluBackward(Z, dA)
    elif activation == "tanh":
        A= tanh(Z)
        dZ = tanhBackward(Z, dA)
    elif activation == "sigmoid":
        A = sigmoid(Z)
        dZ = sigmoidBackward(Z, dA)
    dA_prev = np.dot(W.T, dZ)
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    return dA_prev, dW, db

def linearActivationForward(A_prev, W, b, activation):
    linear_cache = {
        "A_prev": A_prev,
        "W": W,
        "b": b
        }
    Z = np.dot(W, A_prev) + b
    activation_cache = {
        "Z": Z,
        "activation": activation
        }
    if activation == "tanh":
        A = tanh(Z)
    if activation == "sigmoid":
        A = sigmoid(Z)
    if activation == "relu":
        A = relu(Z)
    caches = (linear_cache, activation_cache)
    return A, caches

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return A
def sigmoid(Z):
    A = 1 / (1+np.exp(-Z))
    return A
def relu(Z):
    Z[Z < 0] = 0 #each value beneath 0 will have activation 0, but Z otherwise
    A = Z
    return A

def tanhBackward(Z, dA):
    dZ = (1-np.power(tanh(Z),2)) * dA
    return dZ
def sigmoidBackward(Z, dA):
    dZ = (sigmoid(Z) * (1 - sigmoid(Z)))* dA
    return dZ
def reluBackward(Z, dA):
    Z[Z<0] = 0
    Z[Z>= 0] = 1
    dZ = Z * dA
    return dZ

def loadDataCSV(path):
    file = open(path)
    csvreader = csv.reader(file)
    dataset = []
    count = 0
    for row in csvreader:
           if count == 0:
               count +=1 
               continue 
           if count == 1:
               count = 2
               dataset = np.array([row])
               continue
           row = np.array([row])
           dataset = np.concatenate((dataset,row), axis = 0)
    
    trainSet = dataset[:int(len(dataset) * 0.7)].T
    testSet = dataset[int(len(dataset) * 0.7):].T
    Ytrain = trainSet[-1].astype(np.float).reshape(1, trainSet.shape[1])
    Ytest = testSet[-1].astype(np.float).reshape(1, testSet.shape[1])
    trainSet  = np.delete(trainSet, -1, 0)
    testSet = np.delete(testSet, -1, 0)
    Xtrain = trainSet.astype(np.float)
    Xtest = testSet.astype(np.float)
    means = 1 / Xtrain.shape[0] * np.sum(Xtrain, axis = 1, keepdims = True)
    Xtrain = Xtrain /means
    Xtrain = Xtrain / (np.sqrt(1/Xtrain.shape[0] * np.sum(Xtrain, axis =1, keepdims = True)))
    Xtest = Xtest / means
    Xtest = Xtest / (np.sqrt(1/Xtrain.shape[0] * np.sum(Xtrain, axis =1, keepdims = True)))
 


    return Xtrain, Ytrain, Xtest, Ytest

if __name__ == "__main__":
    main()