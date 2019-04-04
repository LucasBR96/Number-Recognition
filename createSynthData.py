import numpy
import csv
import os
import random

os.chdir(r'C:\Users\Lucas\Desktop\Coding\Actual Coding\python\Fineshed Projects\Machine Learning')

#----------------------------------------------------------------------------------------------------------------------------------
# The mnist dataset is famous for being the "Hello Word" version of AI and machine learning. The starting point to master this area.
# After training a few times with the original dataset mnist I decided to add some flavor to it.
# 
# In the course of Machine Learning of the university of stanford, professor Andrew Ng introduces on week 9 the concept of Data Synthetis.
# It means, in order to expand your dataset, you can create new data from the original. But you have to respect some rules.
# 
# This is all what this script is about.
# You see, in the original set,there are 70000 white handwritten digits (from 0 to 9) on a black background.
# The code below wil copy the original data, but swap the colors, and then both datasets will then be shuffled and merged into a
# Larger file like a poker card deck.
#
# This behemonth of a file will then be used to train a neural network that will be smarter than one trained with original mnist
# digits. ( I Hope so )
#---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------------------------------------
    # data will be read, copied, shuffled and written by batches of 10000
    #----------------------------------------------------------------------------------------------------------------------------
    f1 = open("mnist_train.csv")
    rawTrainData = csv.reader(f1)

    f2 = open("mnist_test.csv")
    rawTestData = csv.reader(f2)

    indexes = list(range(20000))
    random.shuffle(indexes)

    for batch in range(7):
    
        #----------------------------------------------------------------------------------------------------------------------
        # Reading
        X = []
        Y = []

        XSynth = []
        YSynth = []
        
        #batches from mnist_train
        if batch < 6:
            i = 0
            while i < 10000:
                row = rawTrainData.__next__()
                Y.append(row[0])
                YSynth.append(row[0])

                X.append(tuple(row[1:]))
                i += 1
        
        #batches from mnist_test
        else:
            i = 0
            while i < 10000:
                row = rawTestData.__next__()
                Y.append(row[0])
                YSynth.append(row[0])

                X.append(tuple(row[1:]))
                i += 1

        #-----------------------------------------------------------------------------------------------------------------
        # Syntesis
        i = 0
        for row in X:
            XSynth.append(tuple(map(lambda x: str( 255 - int(x)), row)))   

            print("row {} synthetizied".format(10000*batch + i + 1))
            i += 1 
        
        #-----------------------------------------------------------------------------------------------------------------
        # Shuffle and writing
        with open("mnist_data.txt", mode = 'a') as f:

            for idx in indexes:

                if idx >= 10000:
                    x = XSynth[10000 - idx]
                    y = YSynth[10000 - idx]
                
                else:
                    x = X[idx]
                    y = Y[idx]                        

                f.write(y + " " + " ".join(x) + '\n')

    f1.close()
    f2.close()
