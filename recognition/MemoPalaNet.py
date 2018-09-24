import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd, init
import resnet
import numpy as np
import CelebA

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


# Blocks
class ResBlock(nn.HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(residual+x, act_type='relu')
        return x
    
class ResNet(nn.HybridBlock):

    def __init__(self, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 512], **kwargs):
        super(ResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.features = nn.HybridSequential(prefix='')
        self.features.add(_conv3x3(channels[0], 1, 0))
        self.channels = channels
        self.layers = layers
        for i, num_layer in enumerate(layers):
            stride = 2 if i == 1 else 1
            self.features.add(self._make_layer(num_layer, channels[i+1],
                                               stride, i+1, in_channels=channels[i]))
        self.decay_factor = 2

    def _make_layer(self, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(ResBlock(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(ResBlock(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        feature = self.features(x)

        return feature

class DatasetInfo():
    def __init__(self, dataset_size, cls_num, cls_data_num):
        self.cls_num = cls_num
        # data number in each class
        self.cls_data_num = self.cls_data_num
        # the sum value of position of data in each class
        self.cls_pos = nd.zeros((cls_num, 3))
        # the center position of each data
        self.data_pos = nd.zeros((dataset_size[0], 3))

class MemoPalaNet(nn.HybridBlock):
    """
    Memory palace network for image classification
    dataset_size: size of data set: (N, C, H, W)
    cls_num: number of class
    cls_data_num: list of number of data in each class
    batch_size: batch_size of training data
    ctx: context
    """
    def __init__(self, dataset_size, batch_size, ctx, **kargs):
        super(MemoPalaNet, self).__init__()
        self.ctx = ctx
        self.resnet = ResNet()
        self.dataset_size = dataset_size

        self.pala_size = [0, 0, 0]
        self.pala_size[0] = self.resnet.channels[-1]
        self.pala_size[1] = dataset_size[2] // self.resnet.decay_factor
        self.pala_size[2] = dataset_size[3] // self.resnet.decay_factor

        self.pool_size = 3
        self.batch_size = batch_size
        self.avg_pool_3D = nn.AvgPool3D(pool_size=self.pool_size, strides=1, padding=1)
        self.pos_kernel = self.get_pos_kernel(self.pala_size)
        self.eps = 1e-8

    def initialize(self, data_loader, **kargs):
        super(MemoPalaNet, self).initialize(**kargs)
        """
        Overload the initialize method, it go through all the data in data_loader,
        and maintain the data's positions in memory palace and class information of dataset.
        """
        """
        cls_info[i] maintain the current number of data in class i,
        positions of all the data of class i, and the center pos of class i
        """
        self.cls_info = []
        for i in range(data_loader.cls_num+1):
            self.cls_info.append([0, 0, 0])

        for i in range(data_loader.cls_num+1):
            if data_loader.cls_size_list[i] != 0:
                self.cls_info[i][1] = nd.zeros((data_loader.cls_size_list[i], 3), ctx=self.ctx)
                self.cls_info[i][2] = nd.zeros(3, ctx=self.ctx)

        # data_pos maintain the positions of all the data in dataset
        self.data_pos = nd.zeros((data_loader.data_num+1, 3), ctx=self.ctx)
        # update the parameters using the given data loader
        for data, class_id, img_id in data_loader:
            data = data.copyto(self.ctx)
            data_feature = self.resnet(data)
            # get the center pos and max energy ratio of current batch of data
            data_pos, max_energy = self.get_data_info(data_feature)

            #for each data in current batch, update the class info
            for i in range(data.shape[0]):
                cur_cls_id = class_id[i].asscalar()
                cur_cls_data_num = self.cls_info[cur_cls_id][0]
                self.cls_info[cur_cls_id][1][cur_cls_data_num] = data_pos[i]
                self.data_pos[img_id[i].asscalar()] = data_pos[i]
                self.cls_info[cur_cls_id][0] += 1

    def get_pos_kernel(self, size):
        kernel = [[]]*size[0]

        for i in range(size[0]):
            kernel[i] = [[]]*size[1]
            for j in range(size[1]):
                kernel[i][j] = [[]]*size[2]
                for k in range(size[2]):
                    kernel[i][j][k] = [i, j, k]

        kernel = nd.array(kernel, ctx=self.ctx)
        return kernel

    def get_data_info(self, x):
        y = nd.expand_dims(x, axis=1)
        y = self.avg_pool_3D(y)
        y = y[:, 0, :, :, :]
        max_energy = nd.max(y, (1, 2, 3)) * self.pool_size ** 3 / nd.sum(x, (1, 2, 3))

        center_pos = nd.zeros((self.batch_size, 3), ctx=self.ctx)
        kernel = self.pos_kernel.expand_dims(axis=0)
        for i in range(3):
            sum_x = nd.sum(x, axis=[1, 2, 3])
            center_pos[:, i] = nd.sum(x * kernel[:, :, :, :, i], axis=[1, 2, 3]) / (sum_x + self.eps)

        return center_pos, max_energy

    def hybrid_forward(self, F, x):
        x = self.resnet(x)
        center_pos, max_energy = self.get_data_info(x)
        


    def normalized_cut_loss(self, out, label):
        center_pos, max_energy = out
        cls_id, img_id = label

        loss1 = 1-max_energy

        self.cls_info[cls_id][1]

if __name__ == '__main__':
    ctx = mx.gpu(2)
    batch_size = 32
    data_loader = CelebA.get_class_data_loader(batch_size=batch_size)

    #import sys
    #sys.exit()
    net = MemoPalaNet(data_loader.dataset_size, batch_size, ctx)
    test_data = nd.ones((32, 3, 120, 100), ctx=ctx)
    net.initialize(data_loader, init=init.Xavier(), ctx=ctx)
    temp = nd.sum(net.cls_info[14])
    
