import pickle
import numpy as np


def GetData(timeSteps, maxSamples):
    cacheFileName = "data" + str(timeSteps) + ".pkl"
    with open(cacheFileName, "rb") as file:
        (trainX, trainY, testX, testY) = pickle.load(file)

    trainX = np.nan_to_num(trainX)
    testX = np.nan_to_num(testX)

    trainX = trainX[:maxSamples]
    trainY = trainY[:maxSamples]

    trainX = trainX[:, :, 0:2]
    testX = testX[:, :, 0:2]

    return trainX, trainY, testX, testY



