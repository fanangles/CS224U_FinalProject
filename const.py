# Sequence Length
kNUM_CHARS = 30 #think about this right now.
#Generating 1-hot:
# get sentence.
# convert character into integer.
# use numpy.onehot to put it into onehot- of size (N, NUM_CHARS)

# Number of units in the LSTM layers
kN_HIDDEN = 512
# Optimization learning rate
kLEARNING_RATE = .01
# All gradients above this will be clipped
kGRAD_CLIP = 100
# How often should we check the output?
kPRINT_FREQ = 1000
# Number of epochs to train the net
kNUM_EPOCHS = 20
# Batch Size
kBATCH_SIZE = 6

kMAX_BATCHES = 100
