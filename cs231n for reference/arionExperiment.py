
# coding: utf-8

# Goal: convert avi to some kind of numpy array
# ============

# In[1]:

import cv2.cv as cv
import cv2


# In[2]:

import numpy as np

# capture = cv.CaptureFromFile('videotest.avi')
# frames = []
# for i in xrange(200):
#     img = cv.QueryFrame(capture)
#     if img==None: break
# #     print type(img)
# #     tmp = cv.CreateImage(cv.GetSize(img),8,3)
# #     cv.CvtColor(img,tmp,cv.CV_BGR2RGB)
#     tmp = img
#     frames.append(np.asarray(cv.GetMat(tmp))) 
# frames = np.array(frames)


try:
    vidFile = cv2.VideoCapture('videotest.avi')
except:
    print "problem opening input stream"
    sys.exit(1)
if not vidFile.isOpened():
    print "capture stream not open"
    sys.exit(1)

nFrames = int(vidFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) # one good way of namespacing legacy openCV: cv2.cv.*
print "frame number: %s" %nFrames
# fps = vidFile.get(cv2.cv.CV_CAP_PROP_FPS)
# print "FPS value: %s" %fps

ret, frame = vidFile.read() # read first frame, and the return code of the function.
print frame.shape
frames = np.zeros((200,)+frame.shape)
for i in xrange(200):  # note that we don't have to use frame number here, we could read from a live written file.
    # cv2.imshow("frameWindow", frame)
    # cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    ret, frames[i] = vidFile.read() # read next frame, get next return code


# In[3]:

# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
# print frames.shape
# plt.subplot(211)
# plt.imshow(frames[100,:,:,:])
# plt.subplot(212)
# plt.imshow(frames[115,:,:,:])


# Obtain VGG_S dataset:
# --------
# 

# ``!wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl`` <-- already run, so removing from notebook for now.

# Data Transformation and Preparation functions:
# ----

# In[4]:

import io
import scipy.misc

def rescale_img(X):
    N,H,W,C = X.shape;
    XOut = np.zeros((N,224,224,C), dtype=np.uint8)
    for i in range(N):
        XOut[i,:,:,:] = scipy.misc.imresize(X[i,:,:,:], (224,224), interp="nearest")
    #im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    #im = im[::-1, :, :]

    return XOut;

framesResized = rescale_img(frames);
# plt.imshow(framesResized[43,:,:,:])
# print framesResized[43,224/2,224/2,:]
# print frames[43,240,320,:]
# plt.imshow(frames[43,:,:,:])

# Swap the axes around for frames:
print framesResized.shape
framesResized = framesResized.swapaxes(1,2).swapaxes(1,3)
print framesResized.shape
del frames


# ## Importing Model

# In[5]:

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX


# In[12]:

def makeNeuralNet():
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
#     net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
#     net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
#     net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
#     output_layer_vgg = net['fc8']
    ini = lasagne.init.HeUniform()
    net['_fc7'] = DenseLayer(net['drop6'], num_units=4096, W=ini)
    net['_drop7'] = DropoutLayer(net['_fc7'], p=0.5)
    net['_fc8out'] = DenseLayer(net['_drop7'], num_units=6, nonlinearity=lasagne.nonlinearities.softmax, W=ini)
    output_layer_driver = net['_fc8out']
    return net['drop6'], output_layer_driver
    


# In[17]:

import pickle
import theano.tensor as T
import theano

NUM_EPOCHS = 200
BATCH_SIZE = 1
LEARNING_RATE = 0.000000000001


model = pickle.load(open('vgg_cnn_s.pkl'))
# CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']

vgg,driver = makeNeuralNet()
# print model['values']
lasagne.layers.set_all_param_values(vgg, model['values'][:12])
# Load VGG CNN S into the vgg net.

X = T.tensor4()
y = T.matrix()

# training output
output_train = lasagne.layers.get_output(driver, X, deterministic=False)
# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(driver, X, deterministic=True)
model_params = lasagne.layers.get_all_params(driver, trainable=True)

sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
# updates = lasagne.updates.adam(cost, model_params, learning_rate=sh_lr)
updates = lasagne.updates.adam(cost, model_params, learning_rate=LEARNING_RATE)

trainFn = theano.function([X, y], [cost, output_train], updates=updates, allow_input_downcast=True)
evalFn = theano.function([X], [output_eval])
# plt.imshow(np.swapaxes(MEAN_IMAGE,0,2).swapaxes(1,0))

#plt.imshow(MEAN_IMAGE)


# In[18]:

THRESHOLD_CORRECT = 0.8
def estCorrectness(output, actual):
    return np.sum(np.sum((output>0.8)==(actual>0.8),  axis=-1)==6)
    
def train_epoch(X, y):
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch = X[idx]
        y_batch = y[idx]
        cost_batch, output_train = trainFn(X_batch, y_batch)
        costs += [cost_batch]
        #output_train is a matrix of probabilities.
        correct += estCorrectness(output_train, y)

    return np.mean(costs), float(correct)/num_samples


def eval_epoch(X, y):
    output_eval = evalFn(X)
    return float(estCorrectness(output_eval, y)) / X.shape[0]


# In[19]:

framesResized = framesResized - MEAN_IMAGE


# In[20]:

# get_ipython().magic(u"run 'InputProcessor.ipynb'")

keyRef = {
 0x80: 0, # 'shift',
 0x1000: 1,#'c',
 0xb00000:2,# 'left',
 0x500000:3, # 'right',
 0x5000000:4, # 'up',
 0xb000000:5,# 'down'}
}
print keyRef
#0x40-control.

def inputToNPArray(line):
    num = int(line.strip(), 16)
    out = [0,0,0,0,0,0]
    if line==0: return out
    for k in keyRef:
        if (num & k) > 0:
            out[keyRef[k]] = 1
    
    return out

def fileToNPArray(filename):
    n = []
    with open(filename) as f:
        for l in f.readlines():
            n.append(inputToNPArray(l));
    print "Input file conversion complete."
    return np.array(n, dtype=np.float32)

# print fileToNPArray("inputtest.txt")# In[21]:

yLabel = fileToNPArray('inputtest.txt')
print yLabel[0]


# In[22]:

data = {
    'X_train': framesResized[:100],
    'y_train': yLabel[:100],
    'X_valid': framesResized[130:150],
    'y_valid': yLabel[130:150],
    'X_test': framesResized[150:],
    'y_test': yLabel[150:]
}


valid_accs, train_accs, test_accs = [], [], []
try:
    for n in range(NUM_EPOCHS):
        train_cost, train_acc = train_epoch(data['X_train'], data['y_train'])
        valid_acc = eval_epoch(data['X_valid'], data['y_valid'])
        test_acc = eval_epoch(data['X_test'], data['y_test'])
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


# In[ ]:

print prob


# In[ ]:


