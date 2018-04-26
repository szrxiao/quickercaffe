import sys, os.path as op
import caffe
from quickercaffe import *

class DarkTeachSwiftNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def backbone(self,nclass,use_bn=True, deploy=False):
        datalayer = self.bottom;
        teacher = DarkNet('teacher', basemodel = self, namespace='s')
        teacher.backbone(use_bn=use_bn)
        yolo_addextra(teacher, nclass,yolostyle=True, use_bn=use_bn)
        blacklist = teacher.fix_params();
        self.set_namespace(None)
        self.set_bottom(datalayer)
        student = SwiftNet('student',basemodel = self)
        student.backbone()
        yolo_addhead(student,nclass, deploy=False,head_cfg = [(512,1,1), (1024,3,8)]*2)
        #yolo_addextra(student,nclass, head_cfg = [(512,1,1), (1024,3,8)]*2)
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

def yolo_ts1(nclass, use_bn):
    name = 'dark-swiftnet'
    trainnet = DarkTeachSwiftNet(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    blacklist=trainnet.backbone(nclass,use_bn=use_bn, deploy=False)
    trainnet.euclideanloss('yolo_4/conv','s_extra_conv21', loss_weight=0.1)
    trainnet.set_conv_params(blacklist=blacklist)
    trainnet.silence('label','s_last_conv','last_conv')
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = SwiftNet(name)
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True, head_cfg=[(512,1,1),(1024,3,8)]*2)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());
        
def convert_model(nclass, teacher_weights,student_weights, init_weights, use_bn=True ):
    name = 'yoloteacher'
    teacher = DarkNet(name)
    init_proto=op.splitext(init_weights)[0]+'.prototxt'
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(teacher,datafolder,nclass,imgsize,deploy=True)
    teacher.set_namespace('s')
    teacher.backbone(use_bn=use_bn)
    yolo_addhead(teacher, nclass, yolostyle=True,use_bn=use_bn,deploy=True, head_cfg=[(1024,1,1), (1024,3,4),(1024,1,1)] )
    with open(init_proto,'w') as fout:
        fout.write(teacher.toproto());
    transfermodel(teacher_weights, init_weights)
    net=caffe.Net('yolotrain.prototxt',caffe.TEST)
    net.copy_from(init_weights)  #, ignore_shape_mismatch=True
    net.copy_from(student_weights)
    net.save(init_weights)
    
if __name__ == "__main__":
    nclass=12
    use_bn=False
    yolo_ts1(nclass, use_bn=use_bn);
    convert_model(nclass, sys.argv[1], sys.argv[2],sys.argv[3],use_bn=use_bn)
