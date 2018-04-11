import os.path as op
import caffe
from quickercaffe import NeuralNetwork,saveproto, yolo_addinput
            
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
        self.conv(125,1,layername='last_conv')
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
    def backbone(self,deploy=False):
        blacklist=[]
        if (deploy==False):
            datalayer = self.bottom;
            blacklist = self.teacher();
            self.set_bottom(datalayer)
        self.student()
        return blacklist;
    def gen_net(self,nclass,deploy=False):
        blacklist = self.backbone(deploy=deploy)
        head_cfg=[(1024,1,1), (1024,3,4), (1024,1,1)]
        scorelayer = 'last_conv2'
        for i,lcfg in enumerate(head_cfg):
            with self.scope('yolo_'+str(i+1)):
                self.conv(lcfg[0], lcfg[1], pad=(lcfg[1]-1)//2, group=lcfg[2])
                self.bnscale()
                self.leakyrelu(0.1)
        nanker = 5
        self.conv(nanker*(nclass+5),1, layername=scorelayer)
        return blacklist

def yolo_ts():
    name = 'yolo-tinydarknet'
    trainnet = TinyDarkNetTS(name)
    nclass = 20
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False,bias=[1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52])
    blacklist=trainnet.backbone(deploy=False)
    yolo_addhead(trainnet,nclass, deploy=False)
    trainnet.euclideanloss('extra_conv21','yolo/conv3x3_3')
    trainnet.set_conv_params(blacklist=blacklist)
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = TinyDarkNetTS(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone(deploy=True)
    yolo_addhead(testnet, nclass, deploy=True)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());

def classification_ts(nclass,imgsize,batchsize,**kwargs):
    name = 'tinydarknet_ts'
    trainnet = TinyDarkNetTS(name)
    trainnet.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",imgsize[0],batchsize=batchsize,new_image_size =(imgsize[1],imgsize[1]),phases=[caffe.TRAIN,caffe.TEST])
    blacklist = trainnet.gen_net(nclass, **kwargs)
    trainnet.set_conv_params(blacklist=blacklist)
    trainnet.euclideanloss('last_conv','last_conv2')
    
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());

if __name__ == "__main__":
    classification_ts(1000,(224,256),(64,50));
