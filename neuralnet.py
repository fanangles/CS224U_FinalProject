import lasagne
import theano
import theano.tensor as T
import numpy as np;
import time


# create Theano variables for input and target minibatch
sentence1 = T.ftensor3('s1')
sentence2 = T.ftensor3('s2')
mask1 = T.bmatrix('m1')
mask2 = T.bmatrix('m2')
target_var = T.ivector('ent')


# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, LSTMLayer, ConcatLayer, DenseLayer, dropout, Conv1DLayer, ReshapeLayer

from const import *


import dataIO

def createNeuralNetwork():
    #(batchsize, sequence length, onehot vector length)
    in1 = InputLayer((None, None, kNUM_CHARS), sentence1)
    in2 = InputLayer((None, None, kNUM_CHARS), sentence2)
    l_mask1=lasagne.layers.InputLayer((None,None), mask1)
    l_mask2=lasagne.layers.InputLayer((None,None), mask2)

    num_LSTM_output = (512)
    lstm1_f = LSTMLayer(in1, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(),
        nonlinearity=lasagne.nonlinearities.tanh,
        cell_init=lasagne.init.Constant(0.),
        hid_init=lasagne.init.Constant(0.), grad_clipping=kGRAD_CLIP,
        backwards=False,
        mask_input=l_mask1,
        only_return_final=True)
    lstm1_b = LSTMLayer(in1, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(),
        nonlinearity=lasagne.nonlinearities.tanh,
        cell_init=lasagne.init.Constant(0.),
        hid_init=lasagne.init.Constant(0.), grad_clipping=kGRAD_CLIP,
        backwards=True,
        mask_input=l_mask1,
        only_return_final=True)

    lstm2_f = LSTMLayer(in2, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(),
        nonlinearity=lasagne.nonlinearities.tanh,
        cell_init=lasagne.init.Constant(0.),
        hid_init=lasagne.init.Constant(0.), grad_clipping=kGRAD_CLIP,
        backwards=False,
        mask_input=l_mask2,
        only_return_final=True)
    lstm2_b = LSTMLayer(in2, num_LSTM_output,
        forgetgate=lasagne.layers.Gate(),
        nonlinearity=lasagne.nonlinearities.tanh,
        cell_init=lasagne.init.Constant(0.),
        hid_init=lasagne.init.Constant(0.), grad_clipping=kGRAD_CLIP,
        backwards=True,
        mask_input=l_mask2,
        only_return_final=True)

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

def loadDriverModelFromFile(filename):
    print "Loading Neural Network Values from File"
    _v = np.load(filename)['model']
    lasagne.layers.set_all_param_values(network, _v)
    if type(filename)==str:
        print "LOADED"
        return;
    args.modelStoreFile.close()
    print "Loaded!"

if args is not None and args.modelStoreFile is not None:
    loadDriverModelFromFile(args.modelStoreFile)

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
train_fn = theano.function([sentence1, sentence2, mask1, mask2, target_var], loss, updates=updates, allow_input_downcast=True)
# train_fn = theano.function([sentence1, sentence2, target_var], loss, updates=updates)
print "HOLY SHIT IT COMPILED TRAINING FUNCTION"



# train network (assuming you've got some training data in numpy arrays)

def makeMask(batch): #list of matricies of variable size.
    m = np.zeros((len(batch), max([b.shape[0] for b in batch])))
    for i,b in enumerate(batch):
        m[i, 0:b.shape[0]] = 1.0
    return m;

for epoch in range(kNUM_EPOCHS):
    loss = 0
    print "HOLY SHIT IT EPOCHS!"
    #for (batch1,m1), (batch2,m2), ys in dataIO.readChunk(kBATCH_SIZE, './snli_1.0/snli_1.0_train.txt'):
    for (batch1,m1), (batch2,m2), ys in dataIO.readChunk(kBATCH_SIZE, './snli_1.0/snli_1.0_dev.txt'):
        # import code
        # code.interact(local=locals())
        # print batch1.shape, batch2.shape, ys.shape
        # loss += train_fn(batch1, batch2, makeMask(batch1), makeMask(batch2), ys)
        loss += train_fn(batch1, batch2, m1, m2, ys)
        print "yeahhh"

    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(training_data)))
   
    vals = lasagne.layers.get_all_param_values(network)
    modelFile = open("modelStore/"+time.strftime("%m%d-%H%M%S")+".pkl", mode="w")    
    np.savez(modelFile, model=vals)
    print ">>>", modelFile.name
    modelFile.close()
    print "Done."


# Save model

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([sentence1, sentence2, mask1, mask2], T.argmax(test_prediction, axis=1))
print(predict_fn(test_data)) #This might need to be changed
