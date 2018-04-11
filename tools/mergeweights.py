import sys, caffe

net=caffe.Net(sys.argv[1],caffe.TEST)
net.copy_from(sys.argv[2], ignore_shape_mismatch=True)
net.copy_from(sys.argv[3], ignore_shape_mismatch=True)
net.save(sys.argv[4])
