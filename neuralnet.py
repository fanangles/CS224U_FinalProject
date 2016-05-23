import lasagne
import theano
import theano.tensor as T
import numpy as np;

# create Theano variables for input and target minibatch
sentence1 = T.tensor3('s1')
sentence2 = T.tensor3('s2')
mask1 = T.matrix('m1')
mask2 = T.matrix('m2')
target_var = T.ivector('ent')



# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, LSTMLayer, ConcatLayer, DenseLayer, dropout, Conv1DLayer, ReshapeLayer



# Sequence Length
NUM_CHARS = 91 #think about this right now.
#Generating 1-hot:
# get sentence.
# convert character into integer.
# use numpy.onehot to put it into onehot- of size (N, NUM_CHARS)

# Number of units in the LSTM layers
N_HIDDEN = 512
# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
PRINT_FREQ = 1000
# Number of epochs to train the net
NUM_EPOCHS = 10
# Batch Size
BATCH_SIZE = 3

def char_to_onehot(char):
    return (ord(char) - 32) % NUM_CHARS

def sentence_to_onehots(sen):
    # sentence_vector = numpy.empty(0)
    sentence_vector = np.zeros((len(sen), NUM_CHARS))
    for i,char in enumerate(sen):
        onehot = char_to_onehot(char)
        sentence_vector[i,onehot] = 1
    return sentence_vector


from collections import defaultdict;

golds = defaultdict(lambda: 0)
golds['neutral'] = 0;
golds['entailment'] = 1;
golds['contradiction'] = 2;


def readData(filename):
    X1 = [];
    X2 = [];
    Y = [];
    print "reading..."
    with open(filename, 'r') as f:
        f.readline();
        for l in f.readlines():
            l=l.strip().split("\t")
            X1.append(sentence_to_onehots(l[3]));
            X2.append(sentence_to_onehots(l[4]));
            Y.append(golds[l[0]]);

    return X1, X2 ,Y


trainX1, trainX2, trainY = readData('./snli_1.0/snli_1.0_train.txt')
print "HOLY SHIT IT READ DATA"


def createNeuralNetwork():
    #(batchsize, sequence length, onehot vector length)
    in1 = InputLayer((None, None, NUM_CHARS), sentence1)
    in2 = InputLayer((None, None, NUM_CHARS), sentence2)
    l_mask1=lasagne.layers.InputLayer((None,None), mask1)
    l_mask2=lasagne.layers.InputLayer((None,None), mask2)

    num_LSTM_output = (512)
    lstm1_f = LSTMLayer(in1, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(), 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=lasagne.init.Constant(0.), 
        hid_init=lasagne.init.Constant(0.), grad_clipping=GRAD_CLIP,
        backwards=False, mask_input=l_mask1, only_return_final=True)
    lstm1_b = LSTMLayer(in1, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(), 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=lasagne.init.Constant(0.), 
        hid_init=lasagne.init.Constant(0.), grad_clipping=GRAD_CLIP,
        backwards=True, mask_input=l_mask1, only_return_final=True)

    lstm2_f = LSTMLayer(in2, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(), 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=lasagne.init.Constant(0.), 
        hid_init=lasagne.init.Constant(0.), grad_clipping=GRAD_CLIP,
        backwards=False, mask_input=l_mask2, only_return_final=True)
    lstm2_b = LSTMLayer(in2, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(), 
        nonlinearity=lasagne.nonlinearities.tanh, 
        cell_init=lasagne.init.Constant(0.), 
        hid_init=lasagne.init.Constant(0.), grad_clipping=GRAD_CLIP,
        backwards=True, mask_input=l_mask2, only_return_final=True)

    network = ConcatLayer([lstm1_f, lstm1_b, lstm2_f, lstm2_b], axis=1) #(NONE-sentencesize by 2048)
    network = ReshapeLayer(network, (-1, 4, num_LSTM_output));
    #(None by 4x512) I think.
    network = Conv1DLayer(network, 20, 3, pad='same',
                                         nonlinearity=leaky_rectify)
    #(20 by 4 by 52)
    network = Conv1DLayer(network, 10, 3, pad='same',
                                         nonlinearity=leaky_rectify)

    network = lasagne.layers.MaxPool1DLayer(network, 4, stride=2)

    network = DenseLayer(dropout(network, 0.5),
                                        128, nonlinearity=leaky_rectify,
                                        W=lasagne.init.Orthogonal())

    network = DenseLayer(dropout(network, 0.5), 3, nonlinearity=softmax)

    return network;


network = createNeuralNetwork();
print "HOLY SHIT IT COMPILED"
# create loss function
prediction = lasagne.layers.get_output(network)
print "HOLY SHIT IT COMPILED OUTPUT"

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)
print "HOLY SHIT IT COMPILED LOSS FUNCTION WHAT"

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                            momentum=0.9)
print "HOLY SHIT IT COMPILED UPDATES"

# compile training function that updates parameters and returns training loss
train_fn = theano.function([sentence1, sentence2, mask1, mask2, target_var], loss, updates=updates)
print "HOLY SHIT IT COMPILED TRAINING FUNCTION"



# train network (assuming you've got some training data in numpy arrays)

def makeMask(batch): #list of matricies of variable size.
    m = np.zeros((len(batch), max([b.shape[0] for b in batch])))
    for i,b in enumerate(batch):
        m[i, 0:b.shape[0]] = 1
    return m;

for epoch in range(NUM_EPOCHS):
    loss = 0
    print "HOLY SHIT IT EPOCHS!"

    for i in xrange(0, len(trainX1), BATCH_SIZE):
        batch1 = trainX1[i:i+BATCH_SIZE];
        batch2 = trainX2[i:i+BATCH_SIZE]; #(NUMBATCH [ (??), NUM_CHARS)]
        ys = trainY[i:i+BATCH_SIZE];
        m1 = makeMask(batch1);
        loss += train_fn(batch1, batch2, makeMask(batch1), makeMask(batch2), ys)
    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(training_data)))

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
print("Predicted class for first test input: %r" % predict_fn(test_data[0]))