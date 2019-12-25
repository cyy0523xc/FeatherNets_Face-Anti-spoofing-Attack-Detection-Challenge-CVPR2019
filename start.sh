#!/bin/bash
# 
# 
# Author: alex
# Created Time: 2019年12月25日 星期三 19时12分51秒
docker rm -f anti-spoofing
docker run -ti --runtime=nvidia --name anti-spoofing \
    -v "$PWD":/anti-spoofing \
    -v /etc/timezone:/etc/timezone \
    -v /etc/localtime:/etc/localtime \
    -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
    -e PYTHONIOENCODING=utf-8 \
    -w /anti-spoofing \
    registry.cn-hangzhou.aliyuncs.com/ibbd/video:cu100-py36-u1804-cv-tf-pytorch \
    bash

