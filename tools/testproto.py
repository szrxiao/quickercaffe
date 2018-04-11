import sys, caffe

net=caffe.Net(sys.argv[1],caffe.TEST)
net.save(sys.argv[2])
