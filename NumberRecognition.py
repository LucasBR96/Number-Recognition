import BaseFunctions
import NeuralNet
import numpy
import os

os.chdir(r'C:\Users\Lucas\Desktop\Coding\Actual Coding\python\Fineshed Projects\Machine Learning')
Nu = NeuralNet.neuralNetwork(784,256,256,10)
#--------------------------------------------------------------------------------------------------------------------
# Time to test the neural network model on the sythetic mnist data

if __name__ == "__main__":

    f = open( "mnist_data.txt" , mode = "r")

    #---------------------------------------------------------------------------------------------------------------
    # Training the data
    #
    # the traingset will have 130000 examples, since it is too heavy to load a 130000*784 array in one variable the data
    # will flow throught batches. 130 batches with 1000 examples each.
    for i in range(130):

        X = []
        Y = []

        for j in range(1000):
            
            TrainingSample = list(map( int , f.readline().split() ))

            # the first element of each row is the answer of the case
            # it will be converged to an array in order to be compared to the output of the neural network.
            y = BaseFunctions.intToRow( TrainingSample[0] , 10 )
            Y.append(y)

            x = TrainingSample[1:] 
            X.append(x)

        # to avoid overflow, the scale of X's values will be reduced.
        # now each element of x will be in the range [0,1]
        # 0 -> pure black
        # 1 -> pure white
        X = numpy.array(X)/255 
        Y = numpy.array(Y)

        C = Nu.Cost( X , Y )
        print( "Cost of Batch {} = {}".format( i + 1 , C) )

        Nu.train( X , Y , 100 )

    #---------------------------------------------------------------------------------------------------------------------
    # testing the network
    # test set has 100000 examples

    count = 0 #number of correct guesses
    for i in range(10000):

        TestSample = list(map( int , f.readline().split() ))

        y = BaseFunctions.intToRow( TestSample[0] , 10 )
        y = numpy.array(y)
        x = numpy.array(TestSample[1:])/255

        prediction = Nu.predict(x)
        if numpy.array_equal( prediction , y ): count += 1

    f.close()
    print(count/10000)
