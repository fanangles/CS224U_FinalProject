
import argparse

parser = argparse.ArgumentParser(description="Produce Video Minibatches From Video and Key Input Files.")
parser.add_argument('videoFileName', 
	help="The input video file name.", 
	nargs=1)
parser.add_argument('keyFileName', 
	type=argparse.FileType('r'), 
	help="The file storing all the key inputs", 
	nargs=1)
parser.add_argument('--batchSize', type=int, 
	default=100, help="Number of frames per batch")
parser.add_argument('outputDirectory', nargs=1,
	help="Directory to place the output batches")
args = parser.parse_args()

print "Extracting from << X:", args.videoFileName[0], ", Y:", args.keyFileName[0].name, " >> --> Output:", args.outputDirectory[0]


#================================#

import cv2.cv as cv
import cv2
import sys
import os
import numpy as np
import cPickle as pickle
import progressbar
import scipy
import scipy.misc
import h5py 
import scipy.signal

keyRef = {
 0x80: 0, # 'shift',
 0x1000: 1,#'c',
 0xb00000:2,# 'left',
 0x500000:3, # 'right',
 0x5000000:4, # 'up',
 0xb000000:5,# 'down'}
}
#0x40-control.
convolution = np.array([[1.],[1.0],[2.],[9.],[2.],[1.],[1.]])/17.0 

def inputToNPArray(line):
    num = int(line.strip(), 16)
    out = [0.5,]
    # if line==0: return out
    # for k in keyRef:
    #     if (num & k) > 0:
    #         out[keyRef[k]] = 1
    if (num&0xb00000)>0 :
        out[0] = 0
    elif (num&0x500000)>0 :
        out[0] = 1.0

    return out

def fileToNPArray(openfile):
    n = []
    with openfile as f:
        for l in f.readlines():
            n.append(inputToNPArray(l));
    print "Input file conversion complete."
    out = np.array(n, dtype="float16") 
    # out = scipy.signal.convolve(out, convolution, mode='same')
    return out;

def generateOutput(vfilename, kfile, odir, batchsize):
    if not os.path.exists(odir):
        os.makedirs(odir);

    try:
        vidFile = cv2.VideoCapture(vfilename)
    except:
        print "problem opening input file"
        sys.exit(1)
    if not vidFile.isOpened():
        print "capture stream not open"
        sys.exit(1)

    nFrames = int(vidFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) # one good way of namespacing legacy openCV: cv2.cv.*
    print "frame number: %s" %nFrames

    yparsed = fileToNPArray(kfile)
    bar = progressbar.ProgressBar(maxval=(nFrames/batchsize + 1), widgets=[progressbar.Bar('=','[',']'),' ',progressbar.Percentage(), ' ', progressbar.ETA(), ' Done: ', progressbar.Counter()])
    bar.start()
    # ret, frame = vidFile.read() # read first frame, and the return code of the function.
    ret=True;
    info = {'batch': 0, 'origin': vfilename}
    batchID = 0
    while batchID*batchsize < nFrames:
        f = h5py.File(os.path.join(odir,str(batchID)+".hdf5"),'w')
        
        minibatchsize = min(batchsize, nFrames - batchsize*batchID)
        dsetx = f.create_dataset("x", (minibatchsize,3,224,224) , dtype='float32') #shuffle=True
        # frames = np.zeros((batchsize, vidFile.get(cv.CV_CAP_PROP_FRAME_HEIGHT), vidFile.get(cv.CV_CAP_PROP_FRAME_WIDTH), 3))
        for i in xrange(minibatchsize):
            #cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
            ret, frame = vidFile.read() # read next frame, get next return code
            frame = scipy.misc.imresize(frame[15:260], (224,224), interp="nearest")/255.0
            frame[26:55,:,:] = 0.0
            cv2.imshow("-", frame)
            cv2.waitKey(delay=1)
            dsetx[i] = frame.swapaxes(0,1).swapaxes(0,2)
            if not ret:
                # frames = frames[:i] #cutoff.
                raise Exception("out of frames.")
                break;
        #store data:
        # info['x'] = frames;
        currstart = batchID*batchsize;
        # info['y'] = yparsed[currstart:currstart + frames.shape[0]]
        #save to disk.
        # fx =  open(os.path.join(odir,str(batchID)+".x.npy"), 'wb')
        # fy =  open(os.path.join(odir,str(batchID)+".y.npy"), 'wb')
        dset = f.create_dataset("y", data=yparsed[currstart:currstart + dsetx.shape[0]])
        # pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)
        # np.save(fx, info['x'])
        f.close()
        #prep next cycle
        batchID = batchID+1
        bar.update(batchID)

    bar.finish()


generateOutput(args.videoFileName[0], args.keyFileName[0], args.outputDirectory[0], args.batchSize)