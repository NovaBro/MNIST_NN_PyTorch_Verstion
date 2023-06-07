import numpy as np
import sys

np.set_printoptions(suppress = True)
np.set_printoptions(threshold = sys.maxsize)

#____________________________________________________________
# These for loops store non image bytes into arrays
trainingINFO = np.array([])
trainingINFOlabel = np.array([])
def STEP1(testIMG1, testLB1):
    for z in range(4):
        first_str = testIMG1.read(4)
        #NOTE: read reads "(size)"bytes  at a time, in this case 4
        #print(type(str(first_str)), "   ", first_str, int.from_bytes(first_str, byteorder = 'big'))
        
        np.append(trainingINFO, int.from_bytes(first_str, byteorder = 'big'))
        #NOTE: from_bytes converts bytes into integers, byteorder = 'big' means read from right to left
    for z in range(2):
        first_str = testLB1.read(4)
        np.append(trainingINFOlabel, int.from_bytes(first_str, byteorder = 'big'))
            #print(first_str, "  ",  int.from_bytes(first_str, byteorder = 'big'))
    #print("pixelArr.shape", pixelArr.shape, pixelArr)
#--------------------------------------------------------------------------------

#______________________________________________________________
# Initiat labelValueSet values, the number for each picture
def STEP2(labelValueSetSize, labelValueSet, testLB):
    for z in range(labelValueSetSize):
        first_str2 = testLB.read(1)
        labelValueSet[z] = int.from_bytes(first_str2, byteorder = 'big')
#--------------------------------------------------------------------------------

#______________________________________________________________
# Converts images to use, controls how many images in dataSet
# takes images and stores it in dataSet in a series of 1-D arrays
def STEP3(dataSetSize, dataSet, testIMG):
    for i in range(dataSetSize):
        #global pixelArr, labelPrt
        pixelValues = np.zeros((784))
        for z in range(28*28):
            first_str = testIMG.read(1)
            pixelValues[z] = int.from_bytes(first_str, byteorder = 'big')

        dataSet[i] = pixelValues    

#--------------------------------------------------------------------------------

#______________________________________________________________
# Turns label arrays into more arrays in [000...1...00] sort of format,
# what the output of the NN should be
def STEP4(labelSetSize, labelSet, labelValueSet):
    
    for i in range(labelSetSize):
        blankLabel = np.array([0,0,0,0,0,0,0,0,0,0])
        labelIndex = labelValueSet[i]
        blankLabel[int(labelIndex)] = 1
        labelSet[i] = blankLabel
    
    print("start up complete")
#--------------------------------------------------------------------------------

