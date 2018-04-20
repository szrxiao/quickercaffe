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
        yolo_addextra(teacher, nclass,yolostyle=True)
        teacher.set_conv_params()        
        blacklist = teacher.fix_params();
        self.set_bottom(datalayer)
        student = SwiftNet('student',basemodel = self, namespace='s')
        student.backbone()
        #yolo_addhead(student,nclass, deploy=False,head_cfg = [(512,1,1), (1024,3,8)]*2)
        yolo_addextra(student,nclass, head_cfg = [(512,1,1), (1024,3,8)]*2)
        return blacklist

def yolo_ts0(nclass):
    name = 'dark-swiftnet'
    trainnet = DarkTeachSwiftNet(name)
    datafolder = 'data'
    imgsize = 416
    trainnet.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",224,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    blacklist=trainnet.backbone(nclass, deploy=False)
    trainnet.euclideanloss('s_yolo_4/conv','extra_conv21/conv')
    trainnet.set_conv_params(blacklist=blacklist)
    trainnet.silence('label','s_last_conv','last_conv')
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
def yolo_ts1(nclass):
    name = 'dark-swiftnet'
    trainnet = DarkTeachSwiftNet(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    blacklist=trainnet.backbone(nclass, deploy=False)
    trainnet.euclideanloss('s_yolo_4/conv','extra_conv21/conv'，loss_weight=0.1)
    trainnet.set_conv_params(blacklist=blacklist)
    trainnet.silence('label','s_last_conv','last_conv')
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
if __name__ == "__main__":
    yolo_ts0(12);
