from collections import defaultdict;
import numpy as np
from const import *;

golds = defaultdict(lambda: 0)
golds['neutral'] = 0;
golds['entailment'] = 1;
golds['contradiction'] = 2;

def char_to_onehot(char):
    return (ord(char) - 32) % kNUM_CHARS

def sentence_to_onehots(sen):
    # sentence_vector = numpy.empty(0)
    sentence_vector = np.zeros((len(sen), kNUM_CHARS))
    for i,char in enumerate(sen):
        onehot = char_to_onehot(char)
        sentence_vector[i,onehot] = 1
    return sentence_vector

def readData(filename):
    #print "reading..."
    with open(filename, 'r') as f:
        f.readline();
        for l in f:
			#print l
			l=l.strip().split("\t")
			yield sentence_to_onehots(l[3]), sentence_to_onehots(l[4]), golds[l[0]]

def readChunk(batchSize, filename):
	X1 = [];
	X2 = [];
	Y = [];
	for x1,x2,y in readData(filename):
		X1.append(x1)
		X2.append(x2)
		Y.append(y)
		if(len(X1) == batchSize):
			yield np.array(X1),np.array(X2),np.array(Y)
			X1 = [];
			X2 = [];
			Y = [];
	#last chunk.
	yield np.array(X1),np.array(X2),np.array(Y)



def main():
	for x,y,z in readData('./snli_1.0/snli_1.0_train.txt'):
		print x,y,z
		break;

if __name__ == "__main__":
    main()
