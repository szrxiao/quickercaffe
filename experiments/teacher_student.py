import os.path as op
import caffe
from quickercaffe import *
            
class DarkTeachSwiftNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def backbone(self,nclass,deploy=False):
        datalayer = self.bottom;
        teacher = DarkNet('teacher', basemodel = self)
        teacher.backbone()
        yolo_addextra(teacher, nclass)
        teacher.set_conv_params()        
        blacklist = teacher.fix_params();
        self.set_bottom(datalayer)
        student = SwiftNet('student',basemodel = self, namespace='s')
        student.backbone()
        yolo_addhead(student,nclass, deploy=False)
        return blacklist

def yolo_ts(nclass):
    name = 'dark-swiftnet'
    trainnet = DarkTeachSwiftNet(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    blacklist=trainnet.backbone(nclass, deploy=False)
    trainnet.euclideanloss('s_yolo_3/conv','yolo_3/conv')
    trainnet.set_conv_params(blacklist=blacklist)
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
'''
def classification_ts(nclass,imgsize,batchsize,**kwargs):
    name = 'tinydarknet_ts'
    trainnet = TinyDarkNetTS(name)
    trainnet.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",imgsize[0],batchsize=batchsize,new_image_size =(imgsize[1],imgsize[1]),phases=[caffe.TRAIN,caffe.TEST])
    blacklist = trainnet.gen_net(nclass, **kwargs)
    trainnet.set_conv_params(blacklist=blacklist)
    trainnet.euclideanloss('last_conv','last_conv2')
    
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
'''
if __name__ == "__main__":
    yolo_ts(20);
