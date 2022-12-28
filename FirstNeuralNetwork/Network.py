import numpy as np
import os
from PIL import Image
from numpy import asarray
import copy
import matplotlib.pyplot as plt

def main():
    imageSize = 60
    amountOfPictures = 200
    sizeHiddenLayer1 = 200
    sizeHiddenLayer2 = 100
    sizeOutputLayer = 1
    learningRate = 5
    iterations = 8000
    trainingExamples, trainingLabels = loadImages("train", imageSize, amountOfPictures)
    testExamples, testLabels = loadImages("test", imageSize, amountOfPictures)
    testExamples, testLabels = hussle(testExamples, testLabels)
    parameters = trainModel(trainingExamples, trainingLabels, sizeHiddenLayer1, sizeHiddenLayer2, sizeOutputLayer, learningRate, iterations)
    trainingPrediction = predict(trainingExamples, parameters)
    trainingAccuracy = computeAccuracy(trainingPrediction, trainingLabels)
    testPrediction = predict(testExamples, parameters)
    testAccuracy = computeAccuracy(testPrediction, testLabels)
    print("trainingAccuracy", trainingAccuracy)
    print("testAccuracy", testAccuracy)



def computeAccuracy(prediction, Y):
    difference = prediction - Y
    goodanswers = np.count_nonzero(difference == 0)
    accuracy = goodanswers / Y.shape[1] * 100
    return accuracy

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

def trainModel(X, Y, sizeHiddenLayer1,sizeHiddenLayer2, sizeOutputLayer, learningRate, iterations):
    parameters = parameter_initiation(X, sizeHiddenLayer1, sizeHiddenLayer2, sizeOutputLayer)
    costArray = []
    iterationsList = []
    showcost = 200
    for i in range(iterations):
        iterationsList.append(i)
        X, Y = hussle(X,Y)
        A3, cache = forward_propagation(X, parameters)
        costArray.append(cost(A3, Y, X))
        if i == showcost:
            plt.title("cost function")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.plot(iterationsList, costArray, color ="red")
            plt.show()
            showcost += 200
        derivatives = backward_propagation(cache, parameters, Y, X)
        parameters = parameter_update(parameters, derivatives, learningRate)
        
    return parameters

def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    A2[A2 > 0.5] = 1
    A2[A2 <= 0.5] = 0
    return A2

def parameter_initiation(X, sizeHiddenLayer1, sizeHiddenLayer2, sizeOutputLayer):
    '''
    takes inputs X, sizeHiddenLayer, sizeOutputLayer and will output
    and object where each key refers to the corresponding matrix
    '''
    W1 = np.random.randn(sizeHiddenLayer1, X.shape[0]) * 0.001
    W2 = np.random.randn(sizeHiddenLayer2, sizeHiddenLayer1) * 0.001
    W3 = np.random.randn(sizeOutputLayer, sizeHiddenLayer2) * 0.001
    b1 = np.zeros((sizeHiddenLayer1, 1))
    b2 = np.zeros((sizeHiddenLayer2, 1))
    b3 = np.zeros((sizeOutputLayer, 1))

    parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3
        }
    return parameters #checked

def forward_propagation(X, parameters):
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = np.tanh(Z2)
    Z3 = np.dot(parameters["W3"], A2) + parameters["b3"]
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2, 
        "A2": A2,
        "Z3": Z3,
        "A3": A3
        }
    return A3, cache #checked

def backward_propagation(cache,parameters, Y, X):
    m = X.shape[1]
    dZ3 = cache["A3"] - Y #checked
    dA2 = np.dot(parameters["W3"].T, dZ3)

    dZ2 = (1-np.power(2, cache["A2"])) * dA2
    dA1 = np.dot(parameters["W2"].T,dZ2) 
    dZ1 = (1 - np.power(2, cache["A1"])) * dA1 #checked
    



    dW1 = 1/m * np.dot(dZ1, X.T)
    dW2 = 1/m * np.dot(dZ2, cache["A1"].T) #checked
    dW3 = 1/m * np.dot(dZ3, cache["A2"].T)
    db1 = np.array(1/m * np.sum(dZ1, axis = 1, keepdims = True)) #checked
    db2 = np.array(1/m * np.sum(dZ2, axis = 1, keepdims = True)) #checked
    db3 = np.array(1/m * np.sum(dZ3, axis = 1, keepdims = True))
    
    derivatives = {
        "dW1": dW1,
        "dW2": dW2,
        "dW3": dW3,
        "db1": db1,
        "db2": db2,
        "db3": db3
        }
    return derivatives #checked

def parameter_update(parameters, derivatives, learningRate):
    W1 = copy.deepcopy(parameters["W1"])
    W2 = copy.deepcopy(parameters["W2"])
    W3 = copy.deepcopy(parameters["W3"])
    b1 = copy.deepcopy(parameters["b1"])
    b2 = copy.deepcopy(parameters["b2"])
    b3 = copy.deepcopy(parameters["b3"])
    dW1 = copy.deepcopy(derivatives["dW1"])
    dW2 = copy.deepcopy(derivatives["dW2"])
    dW3 = copy.deepcopy(derivatives["dW3"])
    db1 = copy.deepcopy(derivatives["db1"])
    db2 = copy.deepcopy(derivatives["db2"])
    db3 = copy.deepcopy(derivatives["db3"])

    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    W3 = W3 - learningRate * dW3
    b1 = b1 - learningRate * db1
    b2 = b2 - learningRate * db2
    b3 = b3 - learningRate * db3
    newParameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3, 
        "b1": b1,
        "b2": b2,
        "b3": b3
        }
    return newParameters

def hussle(X,Y):
    size = X.shape[0]
    Data = np.append(X, Y, axis = 0)
    transposedData = Data.T
    np.random.shuffle(transposedData)
    Data = transposedData.T
    labels = np.array([Data[size]])
    examples = np.delete(Data, size, 0)
    return examples, labels

def cost(A, Y, X):
    m = Y.shape[1]
    losses = -(np.multiply(np.log(A), Y) + np.multiply((1-Y), np.log(1-A)))
    cost = 1/m * np.sum(losses)
    return cost #checked

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z)) 
    return A

if __name__ == "__main__":
    main()
