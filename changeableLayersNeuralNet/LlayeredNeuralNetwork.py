import numpy as np
import csv 
import matplotlib.pyplot as plt
#mwuahhahahahha it works! :)))
def main():
    layerDims = [34, 4, 4,1]
    iterations = 8000
    learningRate = 0.1
    Xtrain, Ytrain, Xtest, Ytest = loadTXT("ionosphere.data")
    parameters = trainmodel(layerDims, Xtrain, Ytrain, iterations, learningRate)
    trainingPrediction = makePrediction(Xtrain, parameters)
    testPrediction = makePrediction(Xtest, parameters)
    print("trainingAccuracy", accuracy(trainingPrediction, Ytrain))
    print("testAccuracy", accuracy(testPrediction, Ytest))

def loadTXT(path):
    dataset = np.loadtxt(path, delimiter = ",", dtype = str)
    dataset = hussle(dataset)
    trainSet = dataset[:int(len(dataset) * 0.7)]
    testSet = dataset[int(len(dataset) * 0.7):]
    Ytrain = trainSet.T[-1].reshape(1, trainSet.shape[0])
    Xtrain = np.delete(trainSet.T, -1, 0).astype(float)
    Ytest = testSet.T[-1].reshape(1, testSet.shape[0])
    Xtest = np.delete(testSet.T, -1, 0).astype(float)
    Ytrain[Ytrain=="g"] = 1
    Ytrain[Ytrain=="b"] = 0
    Ytest[Ytest=="g"] = 1
    Ytest[Ytest=="b"] = 0
    #normalize training data
    XtrainMean = 1/Xtrain.shape[1] * np.sum(Xtrain, axis = 1, keepdims = True)
    Xtrain = Xtrain - XtrainMean
    Xtrainvariance = np.sqrt(1/Xtrain.shape[1] * np.sum(np.power(Xtrain,2),axis = 1, keepdims = True))
    Xtrainvariance[Xtrainvariance == 0] = 0.0000001
    Xtrain = Xtrain / Xtrainvariance
    #normalize test data
    Xtest = Xtest - XtrainMean
    Xtest = Xtest / Xtrainvariance
    return Xtrain, Ytrain.astype(float), Xtest, Ytest.astype(float)

def hussle(dataset):
    np.random.shuffle(dataset)
    return dataset

def accuracy(prediction, Y):
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    difference = Y-prediction
    incorrect = np. count_nonzero(difference)
    false = incorrect / Y.shape[1] * 100
    accuracy = 100 - false
    return accuracy



def trainmodel(layerDims, X, Y, iterations, learningRate):
    costs = []
    iterationlist = []
    showcost = -100
    #parameter initialization (xavier)
    parameters = parameterInitialization(X, layerDims)
    #forward propagation
    for i in range(iterations):
        iterationlist.append(i)
        AL, forward_cache = L_model_forward(X, parameters)
        
        #compute cost (maybe try out L2)
        thiscost = cost(AL, Y)
        costs.append(thiscost)
        #backward propagation (with gradient checking)
        grads = L_model_backward(AL, Y, forward_cache)
        #updating weights
        parameters = updateWeights(grads, parameters, layerDims, learningRate)
        if i == showcost:
            plt.title("cost function")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.plot(iterationlist, costs, color ="red")
            plt.show()
            showcost += 100
    return parameters

def cost(A, Y):
    m = Y.shape[1]
    losses = -(np.multiply(np.log(A), Y) + np.multiply((1-Y), np.log(1-A)))
    cost = 1/m * np.sum(losses)
    return cost #checked


def makePrediction(X, parameters):
    AL, _ = L_model_forward(X, parameters)
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
    return parameters
    
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = int(len(parameters) / 2)
    for l in range(1, L):
        A_prev = A
        A, cache = linearActivationForward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        #linear activation forward returns the next A for the next layer but also returns a cache containing
        #linear (W, A_PREV, b) and activation (Z, "activation" => "relu") cache
        caches.append(cache)
    AL, cache = linearActivationForward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    dAl = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    currentCache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAl, currentCache, "sigmoid")
    #grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    for l in reversed(range(L-1)):
       
        dA = dA_prev_temp
        currentCache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, currentCache, "relu")
        #grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
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
    #just turn into switch
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



#gradient checking because it doesnot work great :)

def gradient_check_n(parameters,gradients,X,Y,epsilon):
    parameter_values, parameter_keys = dictionary_to_vector(parameters)
    grad, grad_keys = dictionary_to_vector(gradients)
    num_parameters = parameter_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
        theta_plus = np.copy(parameter_values)
        theta_minus = np.copy(parameter_values)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus[i] = theta_minus[i] - epsilon
        AL_plus, cache_plus = L_model_forward(X, vector_to_dictionary(theta_plus))
        AL_minus, cache_minus = L_model_forward(X, vector_to_dictionary(theta_minus))
        J_plus[i] = cost(AL_plus, Y)
        J_minus[i] = cost(AL_minus, Y)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    return difference


def dictionary_to_vector(parameters):
    keys = []
    count = 0
    for key in parameters.keys():
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key] * new_vector.shape[0]
        if count ==0:
            theta = new_vector
        else:
            theta = np.concatenate((theta,new_vector), axis = 0)
        count += 1
    return theta, keys

def vector_to_dictionary(theta):
    parameters = {}
    parameters["W1"] = theta[:136].reshape(4,34)
    parameters["b1"] = theta[136:140].reshape(4,1)
    parameters["W2"] = theta[140:156].reshape(4,4) 
    parameters["b2"] = theta[156:160].reshape(4,1)
    parameters["W3"] = theta[160:164].reshape(1,4)
    parameters["b3"] = theta[164:].reshape(1,1)
    return parameters
#conclusion problem gradients

if __name__ == '__main__':
    main()
