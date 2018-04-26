import os.path as op
from quickercaffe import *
from qcmodels import TinyDarkNetV3, DarkNetMS

                
def yolo_darknet(nclass):
    name = 'yolo-darknet'
    #head_cfg=[(1024,3,1),(1024,3,1),(1024,3,1)]  # 0.669 @10k iter, 
    head_cfg=[(1024,1,1),(1024,3,4),(1024,1,1)]   #0.66.0218
    #head_cfg=[(1024,1,1),(1024,3,8),(1024,1,1)]   #0.655064 
    #head_cfg=[(512,1,1),(1024,3,8)]*2  # 0.649049  @10k iter,   @20k iter
    #head_cfg=[(1024,1,1),(1024,3,8)]  # 0.658813
    #head_cfg=[(512,1,1),(1024,3,8)]*3 # 0.641684 @10k iter, 0.665731 @20k iter
    trainnet = DarkNet(name)
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False,yolostyle=True,head_cfg=head_cfg )
    trainnet.set_conv_params()
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = DarkNet(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True,yolostyle=True,head_cfg=head_cfg)  
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
    head_cfg = [(1024,1,1), (1024,3,8), (1024,1,1)] 
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
        

def yolo_swift19(nclass):
    name = 'yolo_swift19'
    trainnet = SwiftNet(name)
    #head_cfg=[(512,1,1),(1024,3,8)]*2         # 54.33
    # head_cfg=[(512,1,1),(1024,3,8),(512,1,1)]       52.9908 
    head_cfg=[(1024,1,1),(1024,3,4),(1024,1,1)]       #  53.54
    datafolder = 'data'
    imgsize = 416
    yolo_addinput(trainnet,datafolder,nclass,imgsize,deploy=False)
    trainnet.backbone()
    yolo_addhead(trainnet,nclass, deploy=False, head_cfg=head_cfg)
    trainnet.set_conv_params()
    with open('yolotrain.prototxt','w') as fout:
        fout.write(trainnet.toproto());
    testnet = SwiftNet(name)    
    yolo_addinput(testnet,datafolder,nclass,imgsize,deploy=True)
    testnet.backbone()
    yolo_addhead(testnet, nclass, deploy=True, head_cfg=head_cfg)  
    with open('test.prototxt','w') as fout:
        fout.write(testnet.toproto());                
if __name__ == "__main__":
    yolo_darknet(20)
    #yolov3(12)
    #yolo_swift19(20)

