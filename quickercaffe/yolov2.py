from qclib import *

def yolo_addinput(net, datafolder, nclass, imgsize, deploy=True, **kwargs):
    if deploy==False:
        transform_param = {'mirror': True, 'crop_size': 224, 'mean_value': [104, 117, 123]}
        box_data_param = {'jitter': 0.2,  'hue': 0.1,    'exposure': 1.5,   'saturation': 1.5,
                'max_boxes': 30,     'iter_for_resize': 80,
                'random_min': imgsize,  'random_max': imgsize,
                'random_scale_min': 0.25,    'random_scale_max':  2,
                'rotate_max': 0,
                'labelmap': op.join(datafolder,'labelmap.txt'),
                }
        tsv_data_param = {'source': op.join(datafolder, 'train.tsv'), 'col_data': 2,  'col_label': 1,
            'batch_size': 16,  'new_height': 256,  'new_width': 256
        }
        net.n['data'], net.n['label'] = L.TsvBoxData(ntop=2, transform_param=transform_param, 
            tsv_data_param=tsv_data_param,
            box_data_param=box_data_param
        ) 
    else:
        net.input([1,3,imgsize,imgsize],layername='data')
        net.input([1,2], layername='im_info')
    net.set_bottom('data')

def yolo_addextra(net, nclass, head_cfg=None, use_bn=True, yolostyle=False, anker_count=5,layername='last_conv'):
    head_cfg = [(1024,1,1), (1024,3,4), (1024,1,1)] if head_cfg is None else head_cfg
    for i,lcfg in enumerate(head_cfg):
        prefix = 'yolo_'+str(i+1) if yolostyle==False else 'extra_conv'+str(i+19)
        if yolostyle==False:
            net.conv(lcfg[0], lcfg[1], pad=(lcfg[1]-1)//2, group=lcfg[2], bias=(not use_bn), prefix=prefix)
        else:
            net.conv(lcfg[0], lcfg[1], pad=(lcfg[1]-1)//2, group=lcfg[2], bias=(not use_bn), layername=prefix)
        if use_bn: net.bnscale(prefix=prefix)
        net.leakyrelu(0.1,prefix=prefix)
        
    net.conv(anker_count*(nclass+5),1, layername=layername,bias=True)
    
def yolo_addhead(net, nclass, deploy=True, yolostyle=False, head_cfg=None,use_bn=True,
    biases = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]) :
    scorelayer  = 'last_conv'
    yolo_addextra(net, nclass, use_bn=use_bn, yolostyle=yolostyle, head_cfg=head_cfg ,layername=scorelayer)
    nanker = len(biases)//2
    scorelayer = net.get_layerobj(scorelayer)
    if deploy==False:
        net.n[net.get_globalname('region_loss')] = L.RegionLoss(scorelayer, net.n['label'],
            classes=nclass,
            coords=4,
            bias_match=True,
            param={'decay_mult': 0, 'lr_mult': 0},
            thresh=0.6,
            biases=biases)
    else:
        net.n[net.get_globalname('bbox')], net.n[net.get_globalname('prob')] = L.RegionOutput(scorelayer, net.n['im_info'],
                ntop=2,
                classes=nclass,
                thresh=0.005, # 0.24
                nms=0.45,
                biases=biases)

