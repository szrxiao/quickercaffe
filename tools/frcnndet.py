#!python2
import os.path as op
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math;
import json;
import base64;
import progressbar;
from datetime import datetime

def createpath( pparts ):
    fpath = op.join(*pparts);
    if not os.path.exists(fpath):
        os.makedirs(fpath);
    return fpath;   
    

def encode_array(nparray):
    shapestr = ",".join([str(x) for x in nparray.shape])
    array_binary = nparray.tobytes();
    b64str =  base64.b64encode(array_binary).decode()
    return ";".join([shapestr,b64str]);

def decode_array(bufferstr) :
    (shapestr,b64str) = [x.strip() for x in bufferstr.split(";")];
    arrayshape = [int(x) for x in shapestr.split(",")];
    array_binary = base64.b64decode(b64str);
    nparray = np.fromstring(array_binary, dtype=np.dtype('float32'));
    return nparray.reshape(arrayshape);

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR);
    except:
        return None;
        
def postfilter(scores, boxes, class_map, max_per_image=100, thresh=0.05, nms_thresh=0.3):
    class_num = scores.shape[1];        #first class is background
    # all detections are collected into:
    # all_boxes[cls] = N x 5 array of detections in (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in  xrange(class_num)]
    # skip j = 0, because it's the background class
    for j in range(1,class_num):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, nms_thresh)
        all_boxes[j] = cls_dets[keep, :]

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1] for j in xrange(1, class_num)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image];
            for j in xrange(1, class_num):
                keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]
    det_results = [];
    for j in xrange(1, class_num):
        for rect in all_boxes[j]:
            crect = dict();
            crect['rect'] = [float(x) for x in list(rect[:4])];
            crect['class'] = class_map[j];
            crect['conf'] = float(rect[4]);
            det_results += [crect];
    return json.dumps(det_results);

class FileProgressingbar:
    fileobj = None;
    pbar = None;
    def __init__(self,fileobj):
        fileobj.seek(0,os.SEEK_END);
        flen = fileobj.tell();
        fileobj.seek(0,os.SEEK_SET);
        self.fileobj = fileobj;
        widgets = ['Test: ', progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell());

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--net', dest='net', help='Network to use ' )
    parser.add_argument('--intsv', required=True,   help='input tsv file for images, col_0:key, col_1:imgbase64')
    parser.add_argument('--colkey', required=False, type=int, default=0,  help='key col index');
    parser.add_argument('--colimg', required=False, type=int, default=1,  help='imgdata col index');
    parser.add_argument('--outtsv', required=False, default="",  help='output tsv file with roi info')
    args = parser.parse_args()
    return args

def _get_image_blob(im,target_size=600, max_size=1000):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im.reshape(1,im.shape[0],im.shape[1],3)
    blob = im.transpose((0,3,1,2))
    return blob, im_scale
    

def im_detect(net, im):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    im_blob, im_scale = _get_image_blob(im)
    iminfo = np.array( [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)


    # reshape network inputs
    net.blobs['data'].reshape(*(im_blob.shape))
    net.blobs['im_info'].reshape(*(iminfo.shape))

    # do forward
    forward_kwargs = {'data': im_blob.astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = iminfo.astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]
    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']
    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    return scores, pred_boxes

def tsvdet(caffemodel, intsv_file, key_idx,img_idx,outtsv_file, **kwargs):
    prototxt = op.splitext(caffemodel)[0] + '.prototxt' if 'proto' not in kwargs else kwargs['proto'];
    cmapfile = op.splitext(caffemodel)[0] + '.labelmap' if 'cmap' not in kwargs else kwargs['cmap'];
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(prototxt) :
        raise IOError(('{:s} not found.').format(prototxt))
    cmap = ['background'];
    with open(cmapfile,"r") as tsvIn:
        for line in tsvIn:
            cmap +=[line.split("\t")[0].strip()];
    count = 0;

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('\n\nLoaded network {:s}'.format(caffemodel));
    with open(outtsv_file,"w") as tsv_out:
        with open(intsv_file,"r") as tsv_in :
            bar = FileProgressingbar(tsv_in);
            for line in tsv_in:
                cols = [x.strip() for x in line.split("\t")];
                if len(cols)> 1:
                    # Load the image
                    im = img_from_base64(cols[img_idx]);
                    # Detect all object classes and regress object bounds
                    scores, boxes = im_detect(net, im)
                    #tsv_out.write('\t'.join([cols[key_idx], encode_array(scores), encode_array(boxes)])+"\n")
                    results = postfilter(scores,boxes, cmap)
                    tsv_out.write(cols[key_idx] + "\t" + results+"\n")
                    count += 1;
                bar.update();
    caffe.print_perf(count);
    return count;

if __name__ == '__main__':
    args = parse_args()
    outtsv_file = args.outtsv if args.outtsv!="" else os.path.splitext(args.intsv)[0]+".eval";
    caffemodel = args.net;

    if args.gpu_id<0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    tsvdet(caffemodel, args.intsv, args.colkey, args.colimg, outtsv_file);
