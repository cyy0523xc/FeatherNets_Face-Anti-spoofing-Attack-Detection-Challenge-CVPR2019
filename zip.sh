#!/bin/bash
# 
# 将项目打包压缩
# Author: alex
# Created Time: 2019年03月11日 星期一 10时38分16秒

# 把最新的源码复制到发布目录
version=
if [ $# = 1 ]; then 
    version="-$1"
fi
date_str=`date -I`
dirname='FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019'

cd ../
if [ ! -d $dirname ]; then
    echo "$PWD: 当前目录错误."
fi
filename=$dirname-"${date_str//-/}$version".zip
if [ -f "$filename" ]; then
    rm -f "$filename"
fi

zip -r "$filename" $dirname \
    -x "*/.git/*" \
    -x "*/.*" \
    -x "*/zip.sh" \
    -x "*/*/*.swp" \
    -x "*/__pycache__/*" 
echo "zip done."

# 发到测试机
ls -alh "$filename"
