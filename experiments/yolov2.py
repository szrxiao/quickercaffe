import os.path as op
from quickercaffe import *
from qcmodels import TinyDarkNetV3, DarkNetMS, WeeNetV0

                
def yolo_darknet(nclass):
    name = 'yolo-darknet'
    head_cfg=[(1024,3,1),(1024,3,1),(1024,3,1)]
    trainnet = DarkNet(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False,head_cfg=head_cfg )
    trainnet.set_conv_params()
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = DarkNet(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True,head_cfg=head_cfg)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());

def yolo_darknetms(nclass):
    name = 'yolo-darknetms'
    head_cfg=[(1024,1,1),(1024,3,8),(1024,1,1)]
    trainnet = DarkNetMS(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False,head_cfg=head_cfg )
    trainnet.set_conv_params()
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = DarkNetMS(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True,head_cfg=head_cfg)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());       
        
def yolo_darknetnp(nclass):
    name = 'yolo-darknetnp'
    head_cfg=[(1024,3,1),(1024,3,1),(1024,3,1)]
    trainnet = DarkNetNP(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False,head_cfg=head_cfg )
    trainnet.set_conv_params()
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = DarkNetNP(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True,head_cfg=head_cfg)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());        
def yolo_tinydarknet():
    name = 'yolo-tinydarknetv3'
    trainnet = TinyDarkNetV3(name)
    nclass = 20
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False,)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False)
    trainnet.set_conv_params()
    with open('train.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = TinyDarkNetV3(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());

def yolov3(nclass):
    name = 'yolov3'
    trainnet = DarkNet53(name)
    #head_cfg=[(512,1,1),(1024,3,1),(512,1,1),(1024,3,1),(512,1,1),(1024,3,1)]
    head_cfg = [(1024,3,1), (1024,3,1), (1024,3,1)] 
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False, head_cfg=head_cfg)
    trainnet.set_conv_params()
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = DarkNet53(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True, head_cfg=head_cfg)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());        
        

def yolo_weenet(nclass):
    name = 'yolov3'
    trainnet = WeeNetV0(name)
    head_cfg=[(512,1,1),(1024,3,1),(512,1,1),(1024,3,1),(512,1,1),(1024,3,1)]
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False, head_cfg=head_cfg)
    trainnet.backbone()
    yolo_addreshead(trainnet,nclass, deploy=False)
    trainnet.set_conv_params()
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = WeeNetV0(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True, head_cfg=head_cfg)
    testnet.backbone()
    yolo_addreshead(testnet, nclass, deploy=True)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());                
if __name__ == "__main__":
    yolov3(12)
