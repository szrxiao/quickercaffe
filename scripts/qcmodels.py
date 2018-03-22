from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op
from quickercaffe import NeuralNetwork,saveproto

           
class TinyDarkNetTS(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,group=1,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks-1)//2,group=group,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def teacher(self):
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
        self.conv(1024,1, layername='extra_conv19')
        self.bn(layername='extra_conv19/bn')
        self.scale(layername='extra_conv19/scale')
        self.leakyrelu(0.1,layername='extra_conv19/leaky')
        self.conv(1024,3, pad=1, group=4, layername='extra_conv20')
        self.bn(layername='extra_conv20/bn')
        self.scale(layername='extra_conv20/scale')
        self.leakyrelu(0.1,layername='extra_conv20/leaky')
        self.conv(1024,1, layername='extra_conv21')
        self.bn(layername='extra_conv21/bn')
        self.scale(layername='extra_conv21/scale')
        self.leakyrelu(0.1,layername='extra_conv21/leaky')
        return self.fix_params();
    def student(self):           
        stages = [(64,2),(128,2),(256,2),(512,2),(1024,4)]
        self.dark_block('t_dark1',3, 32)
        self.maxpool(2,2,layername='t_pool1')
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 't_dark'+str(i+2)+chr(ord('a')+j) if stage[1]>1 else 't_dark'+str(2+1)
                if j%2==0 :
                    self.dark_block(s,1,stage[0]//2)
                else:
                    self.dark_block(s,3,stage[0],group=4,stride=(1 if i==len(stages)-1 or j<stage[1]-1 else 2))
    def backbone(self,deploy=True):
        blacklist=[]
        if (deploy==False):
            datalayer = self.bottom;
            blacklist = self.teacher();
            self.set_bottom(datalayer)
        self.student()
        return blacklist;
    def gen_net(self,nclass,deploy=False):
        blacklist = self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params(blacklist=blacklist)    
            
class DarkNetNP(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks-1)//2,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if j%2==0:
                    self.dark_block(s,3,stage[0], stride=2 if j==stage[1]-1 else 1)
                else:
                    self.dark_block(s+'_1',1,stage[0]//2)
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class TinyDarkNetV3(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,group=1,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3),group=group,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [(64,2),(128,2),(256,2),(512,2),(1024,4)]
        self.dark_block('dark1',3, 32)
        self.maxpool(2,2,layername='pool1')
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+2)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(2+1)
                if j%2==0 :
                    self.dark_block(s,1,stage[0]//2)
                else:
                    self.dark_block(s,3,stage[0],group=4,stride=(1 if i==len(stages)-1 or j<stage[1]-1 else 2))
                    
    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class TinyDarkNetV4(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,group=1,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks-1)//2,group=group,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [(64,3),(128,3),(256,3),(512,3),(1024,5)]
        #stages = [(72,3),(144,3),(288,3),(576,3),(1152,3)]
        self.dark_block('dark1',7, 32,stride=2)
        ngroup =1;
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+2)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(2+1)
                if j%5==0 :
                    self.dark_block(s,3,stage[0]//2, group=ngroup)
                elif j%5==1 :
                    self.dark_block(s,1,stage[0]//2)
                elif j%5==2:
                    self.dark_block(s,3,stage[0],group=ngroup,stride=(1 if i==len(stages)-1 or j<stage[1]-1 else 2))
                elif j%5==3:
                    self.dark_block(s,1,stage[0]//2)
                else:
                    self.dark_block(s,3,stage[0], group=ngroup)
            ngroup= ngroup*2

    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
class TinyDarkNetV5a(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,group=1,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks-1)//2,group=group,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [64,128,256,512]
        #stages = [(72,3),(144,3),(288,3),(576,3),(1152,3)]
        self.dark_block('dark1',7, 32,stride=2)
        for i,nout in enumerate(stages):
            ngroup = max(1,nout//128)
            s = 'dark'+str(i+2)
            self.dark_block(s+'a',1,nout//2)
            self.dark_block(s+'b',3,nout//2, group=ngroup)
            self.dark_block(s+'c',3,nout,group=ngroup,stride=2)
        self.dark_block('dark6a',1,512);
        self.dark_block('dark6b',3,1024,group=8);
        self.dark_block('dark6c',1,512);
        self.dark_block('dark6d',3,1024,group=8);

    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
            
class TinyDarkNetV5(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,group=1,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks-1)//2,group=group,stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def backbone(self):
        stages = [64,128,256,512]
        #stages = [(72,3),(144,3),(288,3),(576,3),(1152,3)]
        self.dark_block('dark1',7, 32,stride=2)
        for i,nout in enumerate(stages):
            ngroup = max(1,nout//128)
            s = 'dark'+str(i+2)
            self.dark_block(s+'a',1,nout//2)
            denseconv = self.dark_block(s+'b',3,nout//2, group=ngroup)
            poolconv = self.maxpool(2,2,layer='pool'+str(i))
            self.set_bottom(denseconv)
            self.dark_block(s+'c',3,nout,group=ngroup,stride=2)
            self.concat(poolconv)
        self.dark_block('dark6a',1,512);
        self.dark_block('dark6b',3,1024,group=8);
        self.dark_block('dark6c',1,512);
        self.dark_block('dark6d',3,1024,group=8);

    def gen_net(self,nclass,deploy=False):
        self.backbone()
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

def test_tinydarknet_ts(nclass):
    name = 'tinydarknet_ts'
    trainnet = TinyDarkNetTS(name)
    testnet = TinyDarkNetTS(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])
            
def test_tinydarknet3(nclass):
    name = 'tinydarknet3'
    trainnet = TinyDarkNetV3(name)
    testnet = TinyDarkNetV3(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])
def test_tinydarknet4(nclass):
    name = 'tinydarknet4'
    trainnet = TinyDarkNetV4(name)
    testnet = TinyDarkNetV4(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])
            
def test_tinydarknet5(nclass):
    name = 'tinydarknet5'
    trainnet = TinyDarkNetV5(name)
    testnet = TinyDarkNetV5(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])
def test_tinydarknet5a(nclass):
    name = 'tinydarknet5'
    trainnet = TinyDarkNetV5a(name)
    testnet = TinyDarkNetV5a(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])
            
if __name__ == "__main__":
    #test_fastdarknet(1000,1.25)
    #test_shuffledarknet(1000,4,1.25)
    #test_fasterdarknet(1000,1.25)
    #test_tinydarknet_ts(1000)
    test_tinydarknet5a(1000)
