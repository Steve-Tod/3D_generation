CUDA_ROOT=/usr/local/cuda-10.0
TF_ROOT=/home/stevetod/jzy/anaconda3/envs/tf_py2/lib/python2.7/site-packages/tensorflow

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') 
# /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow/include

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')      
# /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow


set -e
if [ 'tf_approxmatch_g.cu.o' -ot 'tf_approxmatch_g.cu' ] ; then
    echo 'nvcc'
    $CUDA_ROOT/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
fi
if [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch.cpp' ] || [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch_g.cu.o' ] ; then
    echo 'g++'
    g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $TF_INC -I $CUDA_ROOT/include  -L $CUDA_ROOT/lib64/ -O2
fi
