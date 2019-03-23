from collections import defaultdict;
import numpy as np
from const import *;

golds = defaultdict(lambda: 0)
golds['neutral'] = 0;
golds['entailment'] = 1;
golds['contradiction'] = 2;
dictString = "abcdefghijklmnopqrstuvwxyz,.?!"
charInts = defaultdict(lambda: 0)

for i, char in enumerate(dictString):
    charInts[char] = i



def char_to_onehot(char):
    return charInts[char]

def sentence_to_onehots(sen):
    # sentence_vector = numpy.empty(0)
    sentence_vector = np.zeros((len(sen), kNUM_CHARS))
    for i,char in enumerate(sen):
        onehot = char_to_onehot(char)
        sentence_vector[i,onehot] = 1
    return sentence_vector

def readData(filename):
    print "reading "
    #print "reading..."
    with open(filename, 'r') as f:
        f.readline();
        for l in f:
            l=l.strip().split("\t")
            yield sentence_to_onehots(l[5]), sentence_to_onehots(l[6]), golds[l[0]]

import itertools

def padZeroes(inputList, maxlen, dtype=np.float16):
    result = np.zeros((len(inputList), maxlen, kNUM_CHARS), dtype)
    mask = np.zeros((len(inputList), maxlen), dtype=np.int8);
    for i, row in enumerate(inputList):
        for j, val in enumerate(row):
            result[i,j,:] = val;
            mask[i,j] = 1
    return (result, mask)

def readChunk(batchSize, filename):
    X1 = [];
    X2 = [];
    Y = [];
    maxLen1 = 0;
    maxLen2 = 0;
    for x1,x2,y in readData(filename): #(?,NUM_CHAR)
        X1.append(x1)
        X2.append(x2)
        maxLen1 = max(len(x1), maxLen1)
        maxLen2 = max(len(x2), maxLen2)
        Y.append(y)
        if(len(X1) == batchSize):
            yield (padZeroes(X1, maxLen1),padZeroes(X2, maxLen2),np.array(Y, dtype=np.int32))
            X1 = [];
            X2 = [];
            Y = [];
            maxLen1 = 0;
            maxLen2 = 0
            #last chunk.
    #yield (padZeroes(X1, maxLen1),padZeroes(X2, maxLen2),np.array(Y, dtype=np.int32))
    #yield (np.array(X1), np.array(X2), np.array(Y))


def main():
    batch_num = 0
    for (x,xm),(y,ym),z in readChunk(kBATCH_SIZE,'./snli_1.0/snli_1.0_dev.txt'):
        print x.shape
        batch_num += 1
        print str(batch_num)

		# break;

if __name__ == "__main__":
    main()
