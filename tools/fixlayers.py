import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-i', '--input',  required=True, type=str, help='input prototxt')
    parser.add_argument('-o', '--output', required=True, type=str, help='output prototxt')
    parser.add_argument('-n', '--name', required=True, type=str, help='name of the last fixed layer')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    layer_type = '';
    modify_layer = True;
    newlayer=False;
    name=''
    with open(args.input, 'r') as tsvin, open(args.output,'w') as tsvout:
        for line in tsvin:
            cols = [x.strip() for x in line.split(':')]    
            if  cols[0]=='lr_mult' and type=='Convolution' and modify_layer==True :
                tsvout.write('    lr_mult: 0\n')        #disable learning rate in conv layer
            else:            
                tsvout.write(line)
            if line.startswith('layer '):   #start a new layer
                newlayer=True       #start process layer header
                if name==args.name: modify_layer=False;
            if modify_layer==False: continue;        #stop modify future layer 
            if newlayer and cols[0]== 'name' : 
                name=cols[1].strip('"');
            if newlayer and cols[0]== 'type' : 
                newlayer=False      #stop processing layer header
                type=cols[1].strip('"'); 
                if type=='BatchNorm':
                    tsvout.write("  batch_norm_param {\n    use_global_stats: true\n  }\n")
                if type=='Scale':
                    tsvout.write("  param {lr_mult: 0, decay_mult: 0}\n  param {lr_mult: 0, decay_mult: 0}\n")

