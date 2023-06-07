import InitialStartPytorch as IS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#-----[Settings]-----
learning_rate = 1e-3
batch_size = 400
epochs = 200
#--------------------

testIMG = open('/Users/MNIST Data/train-images-idx3-ubyte', 'rb')
testLB = open('/Users/MNIST Data/train-labels.idx1-ubyte', 'rb')

trainingSize = 5000
testingSize = 5000

dataSetSize = trainingSize + testingSize
dataSet = np.zeros((dataSetSize, 784))

labelValueSetSize = dataSetSize
labelValueSet = np.zeros((labelValueSetSize))

labelSetSize = dataSetSize
labelSet = np.zeros((labelSetSize, 10))

IS.STEP1(testIMG, testLB)
IS.STEP2(labelValueSetSize, labelValueSet, testLB)
IS.STEP3(dataSetSize, dataSet, testIMG)
IS.STEP4(labelSetSize, labelSet, labelValueSet)

testIMG.close()
testLB.close()  

trainingPixels = dataSet[0:trainingSize]
trainingValueLabels = labelValueSet[0:trainingSize]
trainingLabels = labelSet[0:trainingSize]

testingPixels = dataSet[trainingSize:dataSetSize]
testingValuesLabels = labelValueSet[trainingSize:dataSetSize]
testingLabels = labelSet[trainingSize:dataSetSize]

T_training = torch.from_numpy(trainingPixels)
T_testing = torch.from_numpy(testingPixels)

T_trainingLabel = torch.from_numpy(trainingLabels)
T_testingLabel = torch.from_numpy(testingValuesLabels)

#trainingPixels.tofile("selectedTraining.csv", sep=",")
#testingPixels.tofile("selectedTesting.csv", sep=",")  

class CustomDataSet(Dataset):
    def __init__(self, dataSet, labelSet) -> None:
        self.data = dataSet .to(torch.float)
        self.labels = labelSet .to(torch.float)

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#1e-3, 400, 200

trainingOBJ = CustomDataSet(T_training, T_trainingLabel)
testingOBJ = CustomDataSet(T_testing, T_testingLabel)
trainDataLoad = DataLoader(trainingOBJ, batch_size, shuffle=True)
testDataLoad = DataLoader(testingOBJ, batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nerualNetwork = nn.Sequential(
            nn.Linear(28*28,16),
            nn.Sigmoid(),
            nn.Linear(16,16),
            nn.Sigmoid(),
            nn.Linear(16,10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.nerualNetwork(x.to(torch.float32))

loss_fn = nn.MSELoss(reduction="sum")

nerualNetwork1 = NeuralNetwork()
optimizer = torch.optim.SGD(nerualNetwork1.parameters(), learning_rate)

def trainingFunc(dataLoad, model, lossFunc, optimizerFunc):
    model.train()
    for batch, (x,y) in enumerate(dataLoad):
        pred = model(x)
        loss = lossFunc(pred, y)

        loss.backward()
        optimizerFunc.step()
        optimizerFunc.zero_grad()
        loss = loss.item()
        print(loss)

def testingFunc(dataLoad, model):
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataLoad:
            pred = model(X)
            #test = (pred.argmax(1) == y).type(torch.float).sum().item()
            #print(pred)
            test = pred.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    correct /= len(dataLoad.dataset)

    print(f"Accuracy: {(100*correct)}%\n")

for t in range(epochs):
    print(f"epoch: {t}\n")
    trainingFunc(trainDataLoad, nerualNetwork1, loss_fn, optimizer)
    testingFunc(testDataLoad, nerualNetwork1)
#---------