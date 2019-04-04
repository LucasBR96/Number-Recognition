import numpy
import NeuralNet

# Just simple functions to aid the entire process

def moldData(x):

    # takes an array and shapes its data based in the folowing logic
    # if x(i,j) == x.max:
    #   x(i,j) = 1
    # else:
    #   x(i,j) = 0

    x = x/x.max()
    return x.astype(numpy.int32)

def intToRow(x, n):

    result = [0 for i in range(n)]
    result[x] = 1
    return result

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))






