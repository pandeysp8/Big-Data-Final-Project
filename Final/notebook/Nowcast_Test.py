"""
Runs tests for nowcast model
"""
import sys
sys.path.append('src/')

import os
import h5py
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch
from torch.autograd import Variable
from pdb import set_trace as st
from IPython import embed

from torchvision import models as tv
from collections import namedtuple

#from metrics import probability_of_detection,success_rate
#from metrics.histogram import compute_histogram,score_histogram
#from losses import lpips
#from metrics import probability_of_detection,success_rate
#from metrics.lpips_metric import get_lpips

#from readers.nowcast_reader import get_data, read_data

norm = {'scale':47.54,'shift':33.44}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='mse',help='Test data .h5 file')
    parser.add_argument('--model', type=str, help='Path to pretrained model to test or the string "pers" for the persistence model')
    parser.add_argument('--output', type=str, help='Name of output .csv file',default='synrad_test_output.csv')  
    parser.add_argument('--batch_size', type=int, help='batch size for testing', default=32)
    parser.add_argument('--num_test', type=int, help='number of testing samples to use (None = all samples)', default=None)
    parser.add_argument('--crop', type=int, help='crops this many pixels along edge before scoring', default=0)
    args, unknown = parser.parse_known_args()

    return args
#args = parser.parse_args()


args = get_args()
model_file      = args.model
test_data_file  = args.test_data
output_csv_file = args.output
crop            = args.crop

def compute_stats(hits,
                  misses,
                  false_alarms,
                  correct_rejections,
                  partial_hits=0,
                  partial_hit_weight=1.0,
                  partial_misses=0,
                  partial_miss_weight=0.0
                  ):
    """
    Computes scoring statistics based on hits (H), misses (M),
    false alarms (F) and correct rejections (C).
    
    Optionally, partial hits and misses can also be passes, along with
    weights for each.
    
    In the case of binary scoring (no partial hits/misses)
    
    stats = {'n_truth':H+M,
             'n_pred':H+F,
             'hits':H,
             'misses':M,
             'false_alarms':F,
             'correct_rejections':C,
             'pod':H/(H+M),
             'far':F/(F+H),
             'csi':H/(H+M+F),
             'bias':(H+F)/(H+M)}
    
    If partial hits/misses are included, then  
      
        hits  :  hits   + partial_hits * partial_hit_weight
        misses  :  misses + partial_misses * partial_miss_weight
    
    
    Parameters
    ----------
    hits   scalar
       Number of hits in scene   
    misses   scalar
       Number of misses in scene
    false_alarms   scalar
       Number of false alarms in scene
    correct_rejections   scalar
       Number of correct rejections in scene
    partial_hits scalar
       Number of partial hits
    partial_hits_weight scalar
       Weight of partial hits in pod/far/csi calculation
    partial_misses
       Number of partial misses
    partial_missses_weight scalar
       Weight of partial misses in pod/far/csi calculation
       
    Returns
    -------
    scores  dict
       Dictionary containing statistics                    
    """
    H=hits+partial_hits*partial_hit_weight
    M=misses+partial_misses*partial_miss_weight
    F=false_alarms
    C=correct_rejections
    
    n_truth=1.0*(H+M)
    n_pred=1.0*(H+F)
    n_any=1.0*(H+M+F)
    
    pod = 1.0*H/n_truth if n_truth>0 else 1.0
    far = 1.0*F/n_pred if n_pred>0 else 0.0
    csi = 1.0*H/n_any if n_any>0 else 1.0
    bias = n_pred/n_truth if n_truth>0 else 1.0
    
    return {'n_truth':  n_truth,
            'n_pred':  n_pred,
            'hits':    H,
            'misses':  M,
            'false_alarms':F,
            'correct_rejections':C,
            'pod':pod,
            'far':far,
            'csi':csi,
            'bias':bias}


def compute_histogram(truth,pred,bins=255,**kwargs):
    """
    Compares two np.array's of similar dimensions by computing a 2D histogram
    over pixel values. This function is mainly a wrapper of numpy.histogram2d.
    
    The output is a matrix of counts.  The rows correspond to values (or bins)
    in the turth, and the columns correspond to values (or 
    bins) in the prediction.  The value at pixel i,j in the output 
    matrix is the count of how many times a turth value falls in bin i 
    and a predicted value falls in bin j at co-located pixels.
    
    A "perfect" prediction would yield a hitograms with non-zero counts along 
    the diagonal and zero everywhere else. 
    
    Standard forecast statistics can be computed quickly form the histogram
    for multiple thresholds by passing output to score_histogram
    
    Parameters
    ----------
    truth np.array 
       Input array representing truth.  
    pred np.array 
       Input array representing prediction
    bins  (see numpy.histogram2d)
       The bin specification.  Pasted form numpy docs: 
            If int, the number of bins for the two dimensions (nx=ny=bins).
            If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
            If [int, int], the number of bins in each dimension (nx, ny = bins).
            If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
            A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    kwargs  
       Additional arguments passed to numpy.histogram2d.
    
    Returns
    -------
    H    numpy array nx x ny
       Histogram
    rowedges   numpy array 1 x nx+1
       Array of edges along rows
    coledges   numpy array  1 x ny+1
       Array of edges along columns
    
    """
    if pred.shape!=truth.shape:
        raise ValueError('Inputs must have same dimension %s!=%s'%(pred.shape,truth.shape))
    pred = pred.flatten()
    truth = truth.flatten()
    H,rowedges,coledges=np.histogram2d(truth,pred,bins=bins,**kwargs)
    return H,rowedges,coledges
    

def score_histogram(hist, truth_bins, pred_bins, thresholds):
    """
    Computes scoring statistics for multiple thresholds based on a histogram
    (computing using compute_histogram).  
    
    It is assumed that the rows of the histogram correspond to values in the 
    prediction, and columns correspond to values in the truth.
    
    Parameters
    ----------
    hist  numpy array  nx x ny
        Score histogram (from compute_histogram)
    truth_bins  1 x nx+1        
        Values corresponding to the rows of the histogram
    pred_bins  1 x ny+1
        Values corresponding to the columns of the histogram
    thresholds   array or dict
        If array, scoring thresholds used to compute statistics
        If dict,  {label : threshold}.  labels are used as keys in output
    
    Returns
    -------
    scores   dict
        Dictionary of scores for each threshold
    """
    if type(thresholds)!=dict:
        # Make it a dict
        thresholds = {t:t for t in thresholds}
    scores={}
    for label,thres in thresholds.items():
        thres_row=np.argmax(truth_bins>=thres)# argmax will give index of first 1
        thres_col=np.argmax(pred_bins>=thres) # argmax will give index of first 1
        H=np.sum(hist[thres_row:,thres_col:])
        F=np.sum(hist[:thres_row,thres_col:])
        M=np.sum(hist[thres_row:,:thres_col])
        C=np.sum(hist[:thres_row,:thres_col])
        scores[label]={'threshold':thres}
        scores[label].update(compute_stats(H,M,F,C)) 
    return scores
    
def to_scaled_tensor(imA):
    # convert to RGB, scale images and cast to pytorch Tensor
    # expected shape is Nx3xHxW
    # first convet to fake RGB
    imA = np.moveaxis(np.transpose(np.tile(np.expand_dims(imA,axis=-1), (1,1,1,3))),-1,0)
    imA = 2*(imA - 127.5)/255 # conver to the range -1:1
    return torch.FloatTensor(imA)

def get_dist(model, yt, yp, n_out):
    # this will take each time step, convert it to RGB, calculate the distance
    d = np.zeros(n_out)
    for ii in range(n_out):
        truth = to_scaled_tensor(yt[...,ii])
        pred = to_scaled_tensor(yp[...,ii])
        d[ii] = model.forward(truth, pred).cpu().detach().numpy().mean()
    # returns the average over the batch for each time step
    return d

def get_lpips(model, yp,yt,batch_size=32,n_out=12):    
    d = np.zeros(n_out, dtype=np.float32)

    #model = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    # do this in batches and average over all images
    # inputs should Nx3xHxW
    num_batches = 0
    for ii in range(0, yt.shape[0], batch_size):
        start = ii
        stop = np.min([start+batch_size, yt.shape[0]])
        d += get_dist(model, yp[start:stop,...], yt[start:stop,...], n_out)
        num_batches += 1
    d = d/num_batches
    return d
    
    
def get_data(train_data, end=1024, pct_validation=0.2, dtype=np.float32):
    # read data: this function returns scaled data
    # what about shuffling ? 
    train_IN, train_OUT = read_data(train_data, end=end, dtype=dtype) 
    # Make the validation dataset the last pct_validation of the training data
    val_idx = int((1-pct_validation)*train_IN.shape[0])
    
    val_IN = train_IN[val_idx:, ::]
    train_IN = train_IN[:val_idx, ::]

    val_OUT = train_OUT[val_idx:, ::]
    train_OUT = train_OUT[:val_idx, ::]

    return (train_IN,train_OUT,val_IN,val_OUT)


def read_data(filename, rank=0, size=1, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    x_keys = ['IN']
    y_keys = ['OUT']
    s = np.s_[rank:end:size]
    with h5py.File(filename, mode='r') as hf:
        IN  = hf['IN'][s]
        OUT = hf['OUT'][s]
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT

def probability_of_detection(y_true,y_pred,threshold):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_pod))

def success_rate(y_true,y_pred,threshold):
    """
    a.k.a    1 - (false alarm rate)
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    sucr=hits/(hits+false_alarms) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_sucr))

def critical_success_index(y_true,y_pred,threshold):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses+false_alarms) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_csi))

def BIAS(y_true,y_pred,threshold):
    """
    Computes the 2^( mean(log BIAS/log 2) )
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=(hits+false_alarms)/(hits+misses) pow(2)-log-averaged over the D channels
    """
    logbias = tf.math.log(run_metric_over_channels(y_true,y_pred,threshold,_bias))/tf.math.log(2.0)
    return tf.math.pow( 2.0, tf.reduce_mean(logbias))

def run_metric_over_channels(y_true,y_pred,threshold,metric):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    [D,] tensor of metrics computed over each channel
    """
    # permute channels to first dim to work with tf.map_fn
    elems = (tf.transpose(y_true,(3,0,1,2)),
             tf.transpose(y_pred,(3,0,1,2)),
             threshold)
    # Average over channels
    return tf.map_fn(metric,elems,dtype=tf.float32)


def _pod(X):
    """
    Single channel version of probability_of_detection
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    return (hits+1e-6)/(hits+misses+1e-6)


def _sucr(X):
    """
    Single channel version of success_rate
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+1e-6)/(hits+fas+1e-6)

def _csi(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+1e-6)/(hits+misses+fas+1e-6)

def _bias(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+fas+1e-6)/(hits+misses+1e-6)

def _threshold(X,Y,T):
    """
    Returns binary tensors t,p the same shape as X & Y.  t = 1 whereever
    X > t.  p =1 wherever Y > t.  p and t are set to 0 whereever EITHER
    t or p are nan.   This is useful for counts that don't involve correct
    rejections.
    """
    t=tf.math.greater_equal(X, T)
    t=tf.dtypes.cast(t, tf.float32)
    p=tf.math.greater_equal(Y, T)
    p=tf.dtypes.cast(p, tf.float32)
    is_nan = tf.math.logical_or(tf.math.is_nan(X),tf.math.is_nan(Y))
    t = tf.where(is_nan,tf.zeros_like(t,dtype=tf.float32),t)
    p = tf.where(is_nan,tf.zeros_like(p,dtype=tf.float32),p)
    return t,p

model_file= "../models/nowcast/mse_model.h5"

test_data_file = "../data/sample/nowcast_testing.h5"

output_csv_file = "../data/synrad_test_output.csv"

class BaseModel():
    def __init__(self):
        pass;
        
    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True, gpu_ids=[0]):
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], version='0.1'): # VGG using our perceptually-learned weights (LPIPS metric)
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids, version=version)
        print('...[%s] initialized'%self.model.name())
        print('...Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1

        return self.model.forward(target, pred)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))

def dssim(p0, p1, range=255.):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def rgb2lab(in_img,mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(in_img)
    if(mean_cent):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    return img_lab

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)

def tensorlab2tensor(lab_tensor,return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor)*100.
    lab[:,:,0] = lab[:,:,0]+50

    rgb_back = 255.*np.clip(color.lab2rgb(lab.astype('float')),0,1)
    if(return_inbnd):
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1.*np.isclose(lab_back,lab,atol=2.)
        mask = np2tensor(np.prod(mask,axis=2)[:,:,np.newaxis])
        return (im2tensor(rgb_back),mask)
    else:
        return im2tensor(rgb_back)

def rgb2lab(input):
    from skimage import color
    return color.rgb2lab(input / 255.)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


from IPython import embed

#from . import networks_basic as networks
#from .__init__ import tensor2im,voc_ap

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%(version,net)))

            if(not is_train):
                print('Loading model from: %s'%model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        elif(self.model=='net'): # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr

def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out



class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif(num==34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif(num==50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif(num==101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif(num==152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_H=64): # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = l2(tensor2np(tensor2tensorlab(in0.data,to_norm=False)), 
                tensor2np(tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = dssim(1.*tensor2im(in0.data), 1.*tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = dssim(tensor2np(tensor2tensorlab(in0.data,to_norm=False)), 
                tensor2np(tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)

def run_model( x_test ):
    if args.model=='pers':
        print('Using persistence model')
        # only keep the data for persistence
        x_test = x_test[...,11:12] 
        x_test = x_test*norm['scale']+norm['shift']
        y_pred = np.concatenate(12*[x_test], axis=-1) 
        x_test = None # just to release memory ...
        print(f'persistence data : {y_pred.shape}')
    elif args.model=='optflow':
        print('Using optical flow model')
        from src.model.benchmarks import OpticalFlow
        of=OpticalFlow()
        y_pred=of.predict(x_test)
        x_test = None # just to release memory ...
        y_pred = y_pred*norm['scale']+norm['shift']
        print(f'y_pred : {y_pred.shape}')
    else:
        print('loading model')
        model = tf.keras.models.load_model(model_file,compile=False)
        y_pred = model.predict(x_test, batch_size=16, verbose=2)
        x_test = None # just to release memory ...
        # scale predictions back to original scale and mean
        if type(y_pred) == list:
            y_pred = y_pred[0]
        y_pred = y_pred*norm['scale']+norm['shift']
        print(f'y_pred : {y_pred.shape}')
    return y_pred


def ssim(y_true,y_pred,maxVal,**kwargs):
    yt=tf.convert_to_tensor(y_true.astype(np.uint8))
    yp=tf.convert_to_tensor(y_pred.astype(np.uint8))
    s=tf.image.ssim_multiscale(
              yt, yp, max_val=maxVal[0], filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    )
    return tf.reduce_mean(s)

def MAE(y_true,y_pred,dum):
    return tf.reduce_mean(tf.keras.losses.MAE(y_true,y_pred))

def MSE(y_true,y_pred,dum):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))

def run_metric( metric, thres, y_true, y_pred, batch_size):
    result = 0.0
    Ltot = 0.0
    n_batches = int(np.ceil(y_true.shape[0]/batch_size))
    print('Running metric ',metric.__name__,'with thres=',thres)
    for b in range(n_batches):
        start = b*batch_size
        end   = min((b+1)*batch_size,y_true.shape[0])
        L = end-start
        yt = y_true[start:end]
        yp = y_pred[start:end]
        result += metric(yt.astype(np.float32),yp,np.array([thres],dtype=np.float32))*L
        Ltot+=L
    return (result / Ltot).numpy() 

def run_histogram(y_true, y_pred, batch_size=1000,bins=range(255)):
    L = len(bins)-1
    H = np.zeros( (L,L),dtype=np.float64) 
    n_batches = int(np.ceil(y_true.shape[0]/batch_size))
    print('Computing histogram ')
    for b in range(n_batches):
        start = b*batch_size
        end   = min((b+1)*batch_size,y_true.shape[0])
        yt = y_true[start:end]
        yp = y_pred[start:end]
        Hi,rb,cb = compute_histogram(yt,yp,bins)
        H+=Hi
    return H,rb,cb 

def main():
    args = get_args()
    model_file= "../models/nowcast/mse_model.h5"

    test_data_file = "../data/sample/nowcast_testing.h5"

    output_csv_file = "../data/synrad_test_output.csv"

    SIZE=5 # spilt test data into this many chunks to avoid running out of memory
    y_pred = []
    y_test = []
    for i in range(SIZE):
        #x_test, y_test, _, _ = get_data(args.test_data, end=args.num_test, pct_validation=0)
        print(f'get data {i+1} of {SIZE}')
        x_test_i,y_test_i = read_data(test_data_file, rank=i, size=SIZE, end=70, dtype=np.float32)
        print(f'x_test {i+1} of {SIZE} : {x_test_i.shape}')
        y_test_i = y_test_i*norm['scale']+norm['shift'] # unscale
        print('predict')
        y_pred_i = run_model( x_test_i )
        y_test.append(y_test_i)
        y_pred.append(y_pred_i)
        x_test_i=None
    y_pred = np.concatenate(y_pred,axis=0)
    y_test = np.concatenate(y_test,axis=0)
    y_test_i=None
    y_pred_i=None
    
    
    # calculate metrics in batches    
    test_scores_lead = {}
    # Loop over 12 lead times
    model = PerceptualLoss(model='net-lin', net='alex', use_gpu=False)#True, gpu_ids=[1])
    for lead in tqdm(range(12)):
        test_scores={}
        if crop > 0: 
            yt = y_test[:,crop:-crop,crop:-crop,lead:lead+1] # truth data
            yp = y_pred[:,crop:-crop,crop:-crop,lead:lead+1] # predictions have been scaled earlier
        else:
            yt = y_test[...,lead:lead+1] # truth data
            yp = y_pred[...,lead:lead+1] # predictions have been scaled earlier
        test_scores['ssim'] = run_metric(ssim, [255], yt, yp, batch_size=32)
        test_scores['mse'] = run_metric(MSE, 255, yt, yp, batch_size=32)
        test_scores['mae'] = run_metric(MAE, 255, yt, yp, batch_size=32)
        test_scores['lpips'] = get_lpips(model,yp,yt,batch_size=32,n_out=1)[0] # because this is scalar
        
        H,rb,cb=run_histogram(yt,yp,bins=range(255))
        thresholds = [16,74,133,160,181,219]
        scores = score_histogram(H,rb,cb,thresholds)
        for t in thresholds:
            test_scores['pod%d' % t] = scores[t]['pod']
            test_scores['sucr%d' % t] = 1-scores[t]['far']
            test_scores['csi%d' % t] = scores[t]['csi']
            test_scores['bias%d' % t] = scores[t]['bias']
        
        test_scores_lead[lead]=test_scores
    
    print(f'saving to : {output_csv_file}')
    df = pd.DataFrame({k:[v] for k,v in test_scores_lead.items()})
    df.to_csv(output_csv_file)
    #print('Test shape',y_test.shape,'Prediction Shape=',y_pred)
    
    return
    

if __name__=='__main__':
    main()


