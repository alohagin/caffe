#########################################################################
# File Name: tag.sh
# Author: GoYchen
# mail: GoYchen@foxmail.com
# Created Time: Sat 01 Aug 2015 08:24:46 AM EDT
#########################################################################
#!/bin/bash
ctags -R /usr/include/glog/ src/ tools/ include/ ./build/src/caffe/proto/caffe.pb.cc ./build/src/caffe/proto/caffe.pb.h

