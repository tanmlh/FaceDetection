from mxnet import nd, gpu
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.load_parameters('../../Model/Face_Detection/test_mxnet.params', ctx=gpu(1))
x = nd.ones((1, 1, 224, 224), ctx=gpu(1));
