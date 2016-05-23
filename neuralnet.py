import lasagne
import theano
import theano.tensor as T

# create Theano variables for input and target minibatch
sentence1 = T.tensor3('s1')
sentence2 = T.tensor3('s2')
mask1 = T.matrix('m1')
mask2 = T.matrix('m2')
target_var = T.ivector('ent')

# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, LSTMLayer, ConcatLayer, DenseLayer, dropout



# Sequence Length
NUM_CHARS = 36 #think about this right now.
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
NUM_EPOCHS = 50
# Batch Size
BATCH_SIZE = 128

def createNeuralNetwork():
    #(batchsize, sequence length, onehot vector length)
    in1 = InputLayer((None, None, NUM_CHARS), s1)
    in2 = InputLayer((None, None, NUM_CHARS), s2)
    l_mask1=lasagne.layers.InputLayer(shape=(None,None), mask1)
    l_mask2=lasagne.layers.InputLayer(shape=(None,None), mask2)

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

    network = ConcatLayer([lstm1_f, lstm1_b, lstm2_f, lstm2_b], axis=0)
    #(4 by 52) I think.
    network = Conv2DLayer(network, 20, (4, 2), pad='same',
                                         nonlinearity=leaky_rectify)
    #(20 by 4 by 52)
    network = Conv2DLayer(network, 10, (3, 3), pad='same',
                                         nonlinearity=leaky_rectify)

    network = Pool2DLayer(network, (4, 4), stride=2, mode='max')

    network = DenseLayer(dropout(network, 0.5),
                                        128, nonlinearity=leaky_rectify,
                                        W=lasagne.init.Orthogonal())

    network = DenseLayer(dropout(network, 0.5), 3, nonlinearity=softmax)

    return network;


network = createNeuralNetwork();
# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                            momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# train network (assuming you've got some training data in numpy arrays)
for epoch in range(100):
    loss = 0
    for input_batch, target_batch in training_data:
        loss += train_fn(input_batch, target_batch)
    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(training_data)))

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
print("Predicted class for first test input: %r" % predict_fn(test_data[0]))