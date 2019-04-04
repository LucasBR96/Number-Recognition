import numpy
import random
import BaseFunctions


#------------------------------------------------------------------------------------------------------------------------------

class neuralNetwork:

    sigma = 2 #Class variable

    #------------------------------------------------------------------------------------------------------------------------------
    #Constructor
    def __init__(self, *args):
        
        # Iterable of ints, representing the layer configuration.
        # That way< args = (3, 3, 2) means:
        #   3 neurons in input layer
        #   3 neurons in the hidden layer
        #   2 neuron in output layer
        self.values = list(args)

        # Weights and bias  
        self.weights = []
        self.bias = [] 
        
        # Randomly initializing the weights
        i = 1
        while i < len(args):

            a = args[i - 1]
            b = args[i]
            
            # Weight version
            values = [random.uniform(-neuralNetwork.sigma,neuralNetwork.sigma) for x in range(a*b)]
            mat = numpy.array(values).reshape(a,b)
            self.weights.append(mat)

            # Bias version
            values = [random.uniform(-neuralNetwork.sigma,neuralNetwork.sigma) for x in range(b)]
            mat = numpy.array(values).reshape(1,b)
            self.bias.append(mat)
            
            i += 1

    #------------------------------------------------------------------------------------------------------------------------------
    # Generating the activation matrices through the neural network
    def feedFoward(self,X):

        a = X.copy() # To avoid changing the values of the input outside the scope of the method.
        aList = []

        for w,b in zip(self.weights,self.bias):

            z = numpy.add(a@w,b) # Applying layer weight to previous activation and adding bias.
            a = BaseFunctions.sigmoid(z) # Activating z with the sigmoid function.
            aList.append(a)

        return aList
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Calculating output values after feeding through the network
    def predict(self,X):
        
        a = self.feedFoward(X)[-1]
        a = BaseFunctions.moldData(a) # rouding the values.

        return a
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Calculating how far is the net from perfection.
    def Cost(self, X, y, omega = 1):

        a = self.feedFoward(X)[-1]
        a -= y
        m,n = a.shape
        soma = numpy.sum(numpy.multiply(a,a))
        soma += omega * sum( map( lambda K : numpy.sum(numpy.multiply(K,K)) , self.weights ) ) 

        return soma/(2*m)
    
    #------------------------------------------------------------------------------------------------------------------------------
    # Train the Network so it can fit the weights and bias and make accurate predictions on new data
    def train(self,X,y, repeat = 100, k = 1, omega = 1):

        for i in range(repeat):

            grad = self.backProp(X, y, omega)
            self.updateWeights(grad,k)

    #------------------------------------------------------------------------------------------------------------------------------
    # Computing the gradient descent of the cost function.
    # Does a single backProp.    
    def backProp(self, X, y, omega = 1):


        m = X.shape[0]
        aList = [X.copy()]
        aList.extend(self.feedFoward(X))
        n = len(aList) - 1
        
        weightlist = self.weights.copy()
        biasList = self.bias.copy()

        resultWeight = []
        resultBias = []

        i = n
        a = aList[i]
        while i > 0:
            
            
            zPrime = numpy.multiply(a, 1 - a)

            if i == n: 
                delta = (a - y)/m

            else:
                weight = weightlist[i] 
                delta = delta@weight.T
                
            
            delta = numpy.multiply(delta,zPrime)
            a = aList[i - 1]

            weight = weightlist[i - 1]
            deviation = a.T@delta + (omega/m)*weight
            resultWeight.append(deviation)

            biasDeviation = numpy.mean(delta,0)
            resultBias.append(biasDeviation)

            i -= 1

        resultWeight.reverse()
        resultBias.reverse()

        return (resultWeight,resultBias)


    #------------------------------------------------------------------------------------------------------------------------------
    # updates weights and bias based on the gradient descent on back prop.    
    def updateWeights(self, grad, k = 1):
        
        weightGrad,biasGrad =  grad
        for weight,gr in zip(self.weights,weightGrad): weight -= k*gr
        for bias,gr in zip(self.bias,biasGrad): bias -= k*gr

    def __add__(self, other):

        assert(self.values == other.values)

        Nu = neuralNetwork(*self.values)
        Nu.weights = []
        Nu.bias = []

        for sWeight, oWeight in zip( self.weights, other.weights):

            w = numpy.concatenate(sWeight.flatten() , oWeight.flatten())
            selectedWeights = random.choices(w, k = sWeight.size ) 
            Nu.weights.append( numpy.array(selectedWeights).reshape(sWeight.shape) )

        for sBias, oBias in zip( self.bias, other.bias):

            w = numpy.concatenate(sBias.flatten() , oBias.flatten())
            selectedWeights = random.choices(w, k = sBias.size ) 
            Nu.bias.append( numpy.array(selectedWeights).reshape(sBias.shape) )

        return Nu
def test():

    Nu = neuralNetwork(5,3,3)

    #print(*Nu.weights,sep = '\n \n')

    #pseudo dataset to test feedfoward and backprop

    # pseudo X
    N = 50 #number of lines
    low = 10
    high = 30

    X = numpy.array([random.uniform(low,high) for i in range(5*N)]).reshape(N,5)

    #pseudoY
    Y = numpy.array([random.choice([0,1]) for i in range(3*N)]).reshape(N,3)


    print("Neural Network weights")
    print(*Nu.weights, sep = '\n \n')

    print('\n' + "Cost with given weights")
    print(Nu.Cost(X,Y))

    print('\n' + "gradient descent:")
    grad = Nu.backProp(X,Y)
    print(*grad, sep ='\n \n')

    print('\n' + "Updated weights:")
    Nu.updateWeights(grad, k = .5)
    print(*Nu.weights, sep ='\n \n')

    print('\n' + "Updated cost")
    print(Nu.Cost(X,Y))

    print('\n' + "training network ....")
    Nu.train(X,Y,repeat = 1000)

    print('\n' + "Cost after training:")
    print(Nu.Cost(X,Y))


    

    

#test()
