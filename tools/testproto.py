import sys, caffe
from google.protobuf import text_format

def load_net(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return load_net_from_str(all_line)
def load_net_from_str(all_line):
    net_param = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(all_line, net_param)
    return net_param

def calculate_macc(prototxt):
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffe.TEST)
    net_proto = load_net(prototxt)
    macc = []
    ignore_layers = ['BatchNorm', 'Scale', 'ReLU', 'Softmax', 
            'Pooling', 'Eltwise', 'Shift', 'Concat']
    for layer in net_proto.layer:
        if layer.type == 'Convolution' or layer.type=='DepthwiseConvolution':
            assert len(layer.bottom) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            assert len(layer.top) == 1
            output_shape = net.blobs[layer.top[0]].data.shape
            assert len(input_shape) == 4
            assert len(output_shape) == 4
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = output_shape[3] * output_shape[1] * output_shape[2]
            assert layer.convolution_param.kernel_h == 0
            assert layer.convolution_param.kernel_w == 0
            assert len(layer.convolution_param.kernel_size) == 1
            m = m * layer.convolution_param.kernel_size[0] * \
                    layer.convolution_param.kernel_size[0]
            m = m * input_shape[1]
            m = m / layer.convolution_param.group
            macc.append((layer.name, m/1000000.))
        elif layer.type == 'InnerProduct':
            assert len(layer.bottom) == 1
            assert len(layer.top) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            output_shape = net.blobs[layer.top[0]].data.shape
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = reduce(lambda x,y:x*y, input_shape)
            m = m * reduce(lambda x,y:x*y, output_shape)
            macc.append((layer.name, m/1000000.))
        #elif layer.type == 'Scale':
            #assert len(layer.bottom) == 1
            #input_shape = net.blobs[layer.bottom[0]].data.shape
            #m = reduce(lambda x,y:x*y, input_shape)
            #macc = macc + m
        else:
            #assert layer.type in ignore_layers, layer.type
            pass

    return macc
net=caffe.Net(sys.argv[1],caffe.TEST)    
macc = calculate_macc(sys.argv[1])
print('MACC',macc ,sum([x[1] for x in macc]) )    
net.save(sys.argv[2])
