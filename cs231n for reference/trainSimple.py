args = None
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains a convolutional neural network and saves it to file")
    parser.add_argument('modelStoreFile', 
        type=argparse.FileType('r'), nargs='?',
        help="The file storing the saved model params to preload the model with" )
    parser.add_argument('--repl', action='store_true',
        help="Puts the model into repl mode after setting up model.")
    args = parser.parse_args()


import cv2.cv as cv
import cv2

import numpy as np

import io
import scipy.misc

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

import theano.tensor as T
import theano
import theano.compile.nanguardmode
import h5py
import progressbar

import sys

NUM_EPOCHS = 50
LEARNING_RATE = 0.008
DATA_DIR = "/media/xiaonan/NeuralNet Data/"

CORR_THR = 0.5
def estCorrectness(output, actual):
    return np.sum(np.abs(output - actual) < 0.2)
    # return np.sum(np.sum((output>CORR_THR)==actual,  axis=-1)==6)


def makeNeuralNet(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224), input_var=input_var)
    net['bnorm'] = BatchNormLayer(net['input'])
    net['conv1'] = ConvLayer(net['bnorm'], num_filters=96, filter_size=5, stride=2) #96*112*112
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False) #96*37...approx
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, pad=1)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool2'], num_units=1024)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.2)
    net['_fc7'] = DenseLayer(net['drop6'], num_units=256)
    net['_drop7'] = DropoutLayer(net['_fc7'], p=0.2)
    net['_fc8out'] = DenseLayer(net['_drop7'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    output_layer_driver = net['_fc8out']
    return  output_layer_driver, net
    

X = T.tensor4()
y = T.matrix()

print "Constructing Neural Network"
driver, net = makeNeuralNet(X);

def loadDriverModelFromFile(filename):
    print "Loading Neural Network Values from File"
    _v = np.load(filename)['model']
    lasagne.layers.set_all_param_values(driver, _v)
    if type(filename)==str:
        print "LOADED"
        return;
    args.modelStoreFile.close()
    print "Loaded!"

if args is not None and args.modelStoreFile is not None:
    loadDriverModelFromFile(args.modelStoreFile)



output_train = lasagne.layers.get_output(driver, deterministic=False)

output_eval = lasagne.layers.get_output(driver, deterministic=True)

loss = lasagne.objectives.squared_error(output_train, y).mean()

model_params = lasagne.layers.get_all_params(driver, trainable=True)

sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
updates = lasagne.updates.momentum(loss, model_params, learning_rate=sh_lr)

print "Compiling Neural Network"

trainFn = theano.function([X, y], [loss, output_train], updates=updates, allow_input_downcast=True)
evalFn = theano.function([X], [output_eval])

numTrainFiles = 0
numValFiles = 0
numTestFiles = 0

def output_epoch(X,y):
    output_eval = evalFn(X)
    return float(estCorrectness(output_eval[0], y)), float(X.shape[0])

#####################################################################

#      __       __   ______   ______  __    __ 
#     |  \     /  \ /      \ |      \|  \  |  \
#     | $$\   /  $$|  $$$$$$\ \$$$$$$| $$\ | $$
#     | $$$\ /  $$$| $$__| $$  | $$  | $$$\| $$
#     | $$$$\  $$$$| $$    $$  | $$  | $$$$\ $$
#     | $$\$$ $$ $$| $$$$$$$$  | $$  | $$\$$ $$
#     | $$ \$$$| $$| $$  | $$ _| $$_ | $$ \$$$$
#     | $$  \$ | $$| $$  | $$|   $$ \| $$  \$$$
#      \$$      \$$ \$$   \$$ \$$$$$$ \$$   \$$
#                                              
#                                              

def get_files():
    # order = range(1, 30)
    order = [1,]
    random.shuffle(order)
    for folderID in order:
        fnames = os.listdir(DATA_DIR+str(folderID))
        fnames = sorted(fnames)
        for index,fn in enumerate(fnames):
            yield (folderID,index,fn)

def get_train_file():
    for folderID, index, fname in get_files(): 
        if index%8>5: #6,7 are reserved for val and things
            continue
        yield DATA_DIR+str(folderID)+"/"+fname
    yield None

def get_val_file():
    for folderID, index, fname in get_files(): 
        if index%8==7: #6,7 are reserved for val and things
            yield DATA_DIR+str(folderID)+"/"+fname

def get_test_file():
    for folderID, index, fname in get_files(): 
        if index%8==6: #6,7 are reserved for val and things
            yield DATA_DIR+str(folderID)+"/"+fname

def compile_saliency(net):
    inp = net['input'].input_var
    output = lasagne.layers.get_output(driver,deterministic=True)
    max_out = T.max(output, axis=1)
    saliency = theano.grad(max_out.sum(), wrt=inp)
    max_class = T.argmax(output, axis=1)
    return theano.function([inp], [saliency, max_class])
def showImg(imgOrig,sal,max_class,title):
    sal = sal[0]
    max_class = max_class[0]
    sal = sal[::-1].transpose(1,2,0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10),facecolor='w')
    plt.subplot(2,2,1)
    plt.title('input')
    plt.imshow(imgOrig)
    plt.subplot(2,2,2)
    plt.title("abs saliency")
    plt.imshow(np.abs(sal).max(axis=-1),cmap='gray')
    plt.subplot(2,2,3)
    plt.title("pos saliency")
    plt.imshow(np.maximum(0, sal)/sal.max())
    plt.subplot(2,2,4)
    plt.title("neg saliency")
    plt.imshow(np.maximum(0, -sal)/(-sal.min()))
    plt.show()

def runSaliencyAnalysis():
    fn = compile_saliency(net)
    f = h5py.File(get_val_file().next())
    Xs = f['x'][0:1]
    Ys = f['y'][0:1]
    img = Xs[0].swapaxes(0,2).swapaxes(0,1)
    saliency,maxclass = fn(Xs)
    showImg(img,saliency,maxclass,"Saliency")




if __name__== "__main__":
    import random, os
    from multiprocessing import *

    numTrainFiles = sum(1 for _ in get_train_file())
    numValFiles = sum(1 for _ in get_val_file())
    numTestFiles = sum(1 for _ in get_test_file())

    queue_training = JoinableQueue(10)
    sentinel = object()

    def load_training(q):
        for f in get_train_file():
            if f == None:
                break;
            fl = h5py.File(f, mode="r")
            x = np.zeros_like(fl['x'])
            y = np.zeros_like(fl['y'])
            fl['x'].read_direct(x)
            fl['y'].read_direct(y)
            fl.close()
            q.put((x,y), True)
        q.put((None, None), True)

    def makeProgressBar(maxCount, title):
        return progressbar.ProgressBar(maxval=maxCount, widgets=[title+':', progressbar.Bar('=','[',']'),' ',progressbar.Percentage(),' ',progressbar.Counter(),'/',str(maxCount),' ',progressbar.ETA()])

    def train_epoch():
        # p = Process(target=load_training, args=(queue_training,))
        # p.start()
        costs = []
        correct = 0
        total = 0
        bar = makeProgressBar(numTrainFiles, "Train")
        bar.start()
        count = 0;
        for file_name in get_train_file():
            # X,y = queue_training.get(True)
            # queue_training.task_done()
            # if y is None or X is sentinel:
            #     break;
            if file_name==None:
                break;
            f = h5py.File(file_name, 'r')
            X = f["x"]
            y = f["y"]

            cost_batch, output_train = trainFn(X, y)
            # print cost_batch, output_train
            # if math.isnan(output_train[0][0]):
            #     raise Exception("NaN ERROR Caught %d"%i)
            costs += [cost_batch]
            correct += estCorrectness(output_train, y)
            total += y.shape[0]

            count += 1
            bar.update(count)

        bar.finish()
        # p.join()
        return np.mean(costs), float(correct)/float(total)

    def val_epoch():
        correct = 0.0;
        total = 0.0;
        bar = makeProgressBar(numValFiles, "Val")
        bar.start()
        count = 0
        for file_name in get_val_file():
            f = h5py.File(file_name, 'r')
            X = f["x"]
            y = f["y"]
            c,t = output_epoch(X,y)
            correct += c
            total += t
            f.close()
            count += 1
            bar.update(count)

        bar.finish()
        return correct / total


    def test_epoch():
        correct = 0.0;
        total = 0.0;
        bar = makeProgressBar(numTestFiles, "Test")
        bar.start()
        count = 0

        for file_name in get_test_file():
            f = h5py.File(file_name, 'r')
            X = f["x"]
            y = f["y"]
            c,t = output_epoch(X,y)
            correct += c
            total += t
            f.close()
            count += 1
            bar.update(count)

        bar.finish()
        return correct / total


    if args is not None and args.repl:
        from ptpython.repl import embed
        embed(globals(), locals())

    valid_accs, train_accs, test_accs = [], [], []

    try:
        for n in range(NUM_EPOCHS):
            print "Epoch:", n, " in progress..."
            train_cost, train_acc = train_epoch()
            valid_acc = val_epoch()
            test_acc = test_epoch()
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            train_accs += [train_acc]

            if (n+1) % 20 == 0:
                new_lr = sh_lr.get_value() * 0.7
                print "New LR:", new_lr
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

            print "Epoch {0}: Train cost {1}, Train acc {2}, val acc {3}, test acc {4}".format(
                    n, train_cost, train_acc, valid_acc, test_acc)
    except KeyboardInterrupt:
        pass

    print "Training complete. Saving..."
    #STORE THE MODEL
    import time
    vals = lasagne.layers.get_all_param_values(driver)
    modelFile = open("modelStore/"+time.strftime("%m%d-%H%M%S")+".v2.model.pkl", mode="w")
    np.savez(modelFile, model=vals)
    print ">>>", modelFile.name
    modelFile.close()
    print "Done."


# def detect_nan_in(i,node,fn):
#     print "stuff";

# def detect_nan_out(i, node, fn):
#     for output in fn.outputs:
#         if (not isinstance(output[0], np.random.RandomState) and
#             np.isnan(output[0]).any()):
#             print('*** NaN detected ***')
#             print i, node
#             theano.printing.debugprint(node)
#             print('Inputs : %s' % [input[0] for input in fn.inputs])
#             print('Outputs: %s' % [output[0] for output in fn.outputs])
#             break

# def inspect_inputs(i, node, fn):
#     print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

# def inspect_outputs(i, node, fn):
#     print " output(s) value(s):", [output[0] for output in fn.outputs]

# mode = theano.compile.MonitorMode(pre_func=None, post_func=detect_nan_out).excluding('local_elemwise_fusion', 'inplace')
# mode = theano.compile.nanguardmode.NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True, optimizer='default', linker=None)
