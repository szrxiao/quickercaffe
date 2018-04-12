from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op
from qclib import NeuralNetwork

def saveproto(trainnet, testnet, scorelayer, labellayer, nclass, imgsize, batchsize, **kwargs  ):
    testnet.input([1,3,imgsize[0],imgsize[0]])
    testnet.gen_net(nclass, deploy=True, **kwargs)
    with open(testnet.name+'_deploy.prototxt','w') as fout:
        fout.write(testnet.toproto());
    trainnet.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",imgsize[0],batchsize=batchsize,new_image_size =(imgsize[1],imgsize[1]),phases=[caffe.TRAIN,caffe.TEST])
    trainnet.gen_net(nclass, **kwargs)
    trainnet.softmaxwithloss(scorelayer,labellayer,layername='loss')
    trainnet.accuracy(scorelayer,labellayer,layername='accuracy')
    trainnet.accuracy(scorelayer,labellayer, top_k=5, testonly=True, layername='accuracy_top5')
    with open(trainnet.name+'_trainval.prototxt','w') as fout:
        fout.write(trainnet.toproto());

class OneLayer(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def backbone(self):
        self.conv(1024,1)

class VggNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def gen_net(self,nclass,depth=16,deploy=False):
        netdefs ={16: [(64,2),(128,2),(256,3),(512,3),(512,3)], 19: [(64,2),(128,2),(256,4),(512,4),(512,4)] }
        assert (depth in netdefs)
        for i,stagedef in enumerate(netdefs[depth]):
            nout = stagedef[0];
            for j in range(stagedef[1]):
                postfix=str(i+1)+'_'+str(j+1)
                self.convrelu(nout,3,pad=1,postfix=postfix)
            self.maxpool(2,stride=2,layername='pool'+str(i+1))
        self.fcrelu(4096,postfix='6')
        self.dropout(deploy=deploy,postfix='6')
        self.fcrelu(4096,postfix='7')
        self.dropout(deploy=deploy,postfix='7')
        self.fc(nclass,bias=True,postfix='8')
        if deploy==False:
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), blacklist=['fc6','fc7','fc8'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=0.1), whitelist=['fc6','fc7','fc8'] )

class CaffeNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def gen_net(self,nclass,deploy=False):
        self.convrelu(96,11,stride=4,postfix='1')
        self.maxpool(3,layername='pool1',stride=2)
        self.lrn(postfix='1',layername='norm')
        self.convrelu(256,5,pad=2,group=2,postfix='2')
        self.maxpool(3,stride=2,layername='pool2')
        self.lrn(postfix='2',layername='norm')
        self.convrelu(384,3,pad=1,postfix='3')
        self.convrelu(384,3,pad=1,group=2,postfix='4')
        self.convrelu(256,3,pad=1,group=2,postfix='5')
        self.maxpool(3,stride=2,layername='pool5')
        self.fcrelu(4096,postfix='6')
        self.dropout(deploy=deploy,postfix='6')
        self.fcrelu(4096,postfix='7')
        self.dropout(deploy=deploy,postfix='7')
        self.fc(nclass,bias=True,postfix='8')
        if deploy==False:
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), whitelist=['conv1','conv3','fc8'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), bias_filler= dict(type='constant', value=1), whitelist=['conv2','conv4','conv5'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=1), whitelist=['fc6','fc7'] )

class ResNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def standard(self, nin, nout, s, stride ):
        bottom = self.bottom
        with self.scope(s):
            self.conv(nout,3, pad=1,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv(nout,3, pad=1, postfix='2')
            scale2 = self.bnscale(postfix='2')
            self.set_bottom(bottom)
            if nin!=nout:
                self.conv(nout,1, stride=stride,postfix='_expand')
                self.bnscale(postfix='_expand')
            self.eltwise(scale2,layername='')
            self.relu()
        return nout;
    def bottleneck(self, nin, nout, s, stride):
        bottom = self.bottom
        with self.scope(s):
            self.conv(nout,1,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv(nout,3,pad=1, postfix='2')
            self.bnscalerelu(postfix='2')
            self.conv(nout*4,1, postfix='3')
            scale3=self.bnscale(postfix='3')
            self.set_bottom(bottom)
            if nin!=nout*4:
                self.conv(nout*4,1,stride=stride,postfix='_expand')
                self.bnscale(postfix='_expand')
            self.eltwise(scale3,layername='')
            self.relu()
        return nout*4
    def gen_net(self, nclass, depth=18, deploy=False):
        net_defs = {
            10:([1, 1, 1, 1], self.standard),
            18:([2, 2, 2, 2], self.standard),
            34:([3, 4, 6, 3], self.standard),
            50:([3, 4, 6, 3], self.bottleneck),
            101:([3, 4, 23, 3], self.bottleneck),
            152:([3, 8, 36, 3], self.bottleneck),
        }
        assert depth in net_defs.keys(), "net of depth:{} not defined".format(depth)
        nunits_list, block_func = net_defs[depth] # nunits_list a list of integers indicating the number of layers in each depth.
        nouts = [64, 128, 256, 512] # same for all nets
        #add stem
        self.conv(64,7,stride=2,pad=3,layername='conv1')
        self.bnscale(postfix='_conv1')
        self.relu(prefix='conv1')
        self.maxpool(3,2,layername='pool1')
        nin = 64;
        for s in range(4):
            nunits = nunits_list[s]
            for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
                if unit > 1 and nunits > 6:
                    block_prefix = 'res' + str(s+2) + 'b' + str(unit-1)  # layer name prefix
                else:
                    block_prefix = 'res' + str(s+2) + chr(ord('a')+unit-1) # layer name prefix
                stride = 2 if unit==1 and s>0  else 1
                nin = block_func(nin, nouts[s], block_prefix, stride)
        self.avepoolglobal(layername='pool5')
        self.fc(nclass,bias=True,postfix='_'+str(nclass))
        if deploy==False:
            self.set_conv_params()
            

class DarkNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def dark_block(self, s, ks, nout):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if j%2==0:
                    self.dark_block(s,3,stage[0])
                else:
                    self.dark_block(s+'_1',1,stage[0]//2)
            if i<len(stages)-1:
                self.maxpool(2,2,layername='pool'+str(i+1));
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class DarkNet53(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def dark_block(self, s, ks, nout,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3),stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def dark_resblock(self, s, nout):
        with self.scope(s):
            res_bottom = self.bottom
            self.dark_block(s+'1', 1, nout//2)
            self.dark_block(s+'2', 3, nout)
            self.eltwise(res_bottom)
    def backbone(self):
        self.dark_block('dark1',3,32)
        stages = [(64,1),(128,2),(256,8),(512,8),(1024,4)]
        for i,stage in enumerate(stages):
            self.dark_block('dark'+str(i+1)+'s2',3,stage[0],stride=2)
            for j in range(stage[1]):
                self.dark_resblock('dark'+str(i+2)+chr(ord('a')+j),stage[0])
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
class TinyDarkNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def dark_block(self, s, ks, nout):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [(16,1),(32,1),(64,1),(128,1),(256,1),(512,1),(1024,1)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if j%2==0:
                    self.dark_block(s,3,stage[0])
                else:
                    self.dark_block(s+'_1',1,stage[0]//2)
            if i<len(stages)-1:
                self.maxpool(2,1 if i==len(stages)-2 else 2,layername='pool'+str(i+1));
        
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool7')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
class MobileNetV1(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def dw_block(self,  s, nin, nout, stride=1):
        prefix = 'conv'+s+'/dw'
        self.dwconv(nin, 3, stride=stride, pad=1, group=nin, layername=prefix )
        self.bnscale(prefix=prefix)
        self.relu(layername='relu'+s+'/dw')
        prefix = 'conv'+s+'/sep'
        self.conv(nout,1,layername=prefix)
        self.bnscale(prefix=prefix)
        self.relu(layername='relu'+s+'/sep')
        return nout
    def backbone(self):
        #add stem
        nin = 32;   #first stage always 32 out channels
        self.conv(nin,3, pad=1,stride=2,layername='conv1')
        self.bnscale(prefix='conv1')
        self.relu(layername='relu1')
        stages = [(2,128),(2,256),(2,512),(6,1024),(1,1024)]
        for i,stage in enumerate(stages):
            nunit = stage[0]
            nout = stage[1]
            for j in range(1,nunit+1):
                block_prefix = '%d_%d'%(i+2,j) if nunit>1 else str(i+2) # layer name prefix
                stride = 2 if nunit>1 else 1
                if j<nunit:
                    nin = self.dw_block(block_prefix,nin,nout//2)
                else:
                    nin = self.dw_block(block_prefix,nin,nout,stride=stride)
        return self.bottom
    def gen_net(self, nclass, deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
# Relu6 not supported by caffe, use relu instead
class MobileNetV2(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def convbnscale(self,nin,ks,s, group=1,stride=1):
        if group==1:
            self.conv(nin,ks, pad=(ks-1)//2, stride=stride, layername = 'conv'+s)           
        elif group==nin:
            self.dwconv(nin,ks,group=group, pad=(ks-1)//2, stride=stride, layername = 'conv'+s)
        else:
            self.conv(nin,ks,group=group, pad=(ks-1)//2, stride=stride, layername = 'conv'+s)           
        self.bnscale(prefix='conv'+s)
    def convbnscalerelu(self,nin,ks,s,group=1,stride=1):
        self.convbnscale(nin,ks,s,group=group,stride=stride)
        self.relu(layername='relu'+s)
    def bottleneck_block(self,  s, nin, nout, stride=1, factor=1):
        inputlayer = self.bottom
        self.convbnscalerelu(nin*factor, 1, s+'/expand')
        self.convbnscalerelu(nin*factor, 3, s+'/dwise',group=nin*factor, stride=stride)
        self.convbnscale(nout, 1, s+'/linear')
        if stride==1 and nin==nout:
            self.eltwise(inputlayer,layername='block_'+s)
        return nout
    def gen_net(self, nclass, factor=6, deploy=False):
        #add stem
        self.convbnscalerelu(32,3,'1',stride=2)
        self.bottleneck_block('2_1', 32, 16, stride=1, factor=1) 
        self.bottleneck_block('2_2', 16, 24, stride=2, factor=6) 
        self.bottleneck_block('3_1', 24, 24, stride=1, factor=6) 
        self.bottleneck_block('3_2', 24, 32, stride=2, factor=6) 
        self.bottleneck_block('4_1', 32, 32, stride=1, factor=6) 
        self.bottleneck_block('4_2', 32, 32, stride=1, factor=6) 
        self.bottleneck_block('4_3', 32, 64, stride=1, factor=6) 
        self.bottleneck_block('4_4', 64, 64, stride=1, factor=6) 
        self.bottleneck_block('4_5', 64, 64, stride=1, factor=6) 
        self.bottleneck_block('4_6', 64, 64, stride=1, factor=6) 
        self.bottleneck_block('4_7', 64, 96, stride=2, factor=6)
        self.bottleneck_block('5_1', 96, 96, stride=1, factor=6)
        self.bottleneck_block('5_2', 96, 96, stride=1, factor=6)
        self.bottleneck_block('5_3', 96, 160, stride=2, factor=6)
        self.bottleneck_block('6_1', 160, 160, stride=1, factor=6)
        self.bottleneck_block('6_2', 160, 160, stride=1, factor=6)
        self.bottleneck_block('6_3', 160, 320, stride=1, factor=6)
        self.convbnscalerelu(1280,1,'6_4')
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,postfix=str(nclass))
        if deploy==False:
            self.set_conv_params()

class ShuffleNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self,name,**kwargs)
    def shuffle_block(self, nin, nout, s, stride=1, group=3, bn_ratio=0.25):
        bottom = self.bottom
        noutnew = nout if stride==1 else nout-nin
        bn_channels = int(noutnew*bn_ratio)
        with self.scope(s):
            if nin==24:
                self.conv1x1(bn_channels,stride=1, postfix='1')
            else:
                self.conv1x1(bn_channels,stride=1, group=group, postfix='1')
            self.bnscalerelu(postfix='1')
            if nin!=24:
                self.shuffle(group=group)
            self.dwconv(bn_channels,3, group=bn_channels, stride=stride, pad=1, postfix='2')
            self.bnscale(postfix='2')
            self.conv1x1(noutnew,stride=1, group=group, postfix='3')
            rightbranch = self.bnscale(postfix='3')
            if stride==1:
                self.eltwise(bottom, postfix='join')
            else:
                self.avepool(3,2,bottom=bottom)
                self.concat(rightbranch,postfix='join')
        return nout;
    def gen_net(self, nclass, group=3,bn_ratio=0.25,deploy=False):
        nout_cfg = { 1:144, 2:200,3:240,4:272, 8:384 }
        net_stages = [4,8,4]

        assert group in nout_cfg.keys(), "net of group:{} not defined".format(group)
        nout_base = nout_cfg[group];
        #add stem
        nin = 24;   #first stage always 24 out channels
        self.conv3x3(nin,stride=2,postfix='1')
        self.bnscalerelu(postfix='1')
        self.maxpool(3,2,layername='pool1')

        for s in range(3):
            nunits = net_stages[s]
            nout = nout_base * (2**s)
            for unit in range(nunits): # for each unit. Enumerate from 1.
                block_prefix = 'resx' + str(s+3) + chr(ord('a')+unit) # layer name prefix
                stride = 2 if unit==0  else 1
                nin = self.shuffle_block(nin, nout, block_prefix, stride=stride, group=group, bn_ratio=bn_ratio)
        self.avepoolglobal(layername='pool5')
        self.fc(nclass,bias=True,postfix=str(nclass))
        if deploy==False:
            self.set_conv_params()

class SwiftNet(NeuralNetwork):
    def __init__ (self, name, **kwargs ):
        NeuralNetwork.__init__(self, name, **kwargs)
    def convbnrelu(self, s, ks, nout,stride=1,group=1):
        with self.scope(s):
            if stride>1 or group>1:
                self.conv(nout,ks, pad=(ks-1)//2, stride=stride, group=group)
            else:
                self.conv(nout,ks, pad=(ks-1)//2)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def swift_block(self, s, nout,stride=1, group=1):
        self.convbnrelu(s+'1',1,nout/2)
        if stride>1:
            self.convbnrelu(s+'2',3,nout*2,stride=stride, group=group)
        else:
            self.convbnrelu(s+'2',3,nout, group=group)
    def backbone(self):
        stages = [1,1,2,3,2]
        groups = [4,4,8,8,16]
        nout = 64
        self.convbnrelu('swift1',7,nout,stride=2)
        for i,nb in enumerate(stages):
            for j in range(nb):
                s = 'swift'+str(i+2)+chr(ord('a')+j) 
                if j==nb-1 and i!=len(stages)-1:
                    self.swift_block(s,nout,stride=2,group=groups[i])
                else:
                    self.swift_block(s,nout,group=groups[i])
            nout*=2
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
#transfer weights from one model to another (which have different layer name, and the same layer structure)            
def transfermodel(oldproto, newproto):
    oldweights = op.splitext(oldproto)[0]+'.caffemodel'
    newweights = op.splitext(newproto)[0]+'.caffemodel'
    oldmodel = caffe.Net(oldproto, oldweights, caffe.TEST)
    newmodel = caffe.Net(newproto,caffe.TEST)

    for i,layer in enumerate(oldmodel.params.items()):
        for j,weight in enumerate(layer[1]):
            newmodel.params.items()[i][1][j].data[...] = weight.data.copy()
    newmodel.save(newweights)
    
def test_shufflenet(nclass=1000, group=3,bn_ratio=0.25):
    name = 'shuffle'+str(group)
    trainnet = ShuffleNet(name)
    testnet = ShuffleNet(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,50], group=3, bn_ratio=0.25)

def test_resnet(nclass, depth):
    name = 'resnet'+str(depth)
    trainnet = ResNet(name)
    testnet = ResNet(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,50], depth=depth)

def test_darknet(nclass):
    name = 'darknet'
    trainnet = DarkNet(name)
    testnet = DarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])

def test_tinydarknet(nclass):
    name = 'tinydarknet'
    trainnet = TinyDarkNet(name)
    testnet = TinyDarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [448,480],[64,50])
    
def test_caffenet(nclass):
    name = 'caffenet'
    trainnet = CaffeNet(name)
    testnet = CaffeNet(name)
    saveproto(trainnet,testnet, 'fc8','label', nclass, [227,256],[64,50])

def test_vggnet(nclass,depth):
    name = 'vgg'+str(depth)
    trainnet = VggNet(name)
    testnet = VggNet(name)
    saveproto(trainnet,testnet, 'fc8','label', nclass, [227,256],[64,50],depth=depth)

def test_mobilenetv1(nclass):
    name = 'mobilenetv1'
    trainnet = MobileNetV1(name)
    testnet = MobileNetV1(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[128,100])

def test_darknet53(nclass):
    name = 'darknet53'
    trainnet = DarkNet53(name)
    testnet = DarkNet53(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [256,300],[64,50])

def test_mobilenetv2(nclass):
    name = 'mobilenetv2'
    trainnet = MobileNetV2(name)
    testnet = MobileNetV2(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,100])

def test_swiftnet(nclass):
    name = 'SwiftNet'
    trainnet = SwiftNet(name)
    testnet = SwiftNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [256,300],[64,100])
            
if __name__ == "__main__":
    #test_resnet(1000,101);
    #test_resnet(1000,18);
    #test_darknet(1000)
    #test_darknet53(1000)
    test_swiftnet(1000)
    #test_tinydarknet(1000)
    #test_caffenet(1000)
    #test_vggnet(1000,16)
    #test_shufflenet(1000,3)
    #test_mobilenetv2(1000)
    #test_mobilenetv1(1000)
    #transfermodel('mold.prototxt','mnew.prototxt')
