import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd, init
import resnet
import numpy as np
import CelebA
import pickle
import Function

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
    """
    [64, 64, 128, 256, 512]
    """
    def __init__(self, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 1024], **kwargs):
        super(ResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.features = nn.HybridSequential(prefix='')
        self.features.add(_conv3x3(channels[0], 1, 0))
        self.channels = channels
        self.layers = layers
        for i, num_layer in enumerate(layers):
            stride = 2 if i == 1 or i == 2 else 1
            self.features.add(self._make_layer(num_layer, channels[i+1],
                                               stride, i+1, in_channels=channels[i]))
        self.decay_factor = 4

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

class MemoPalaNet(nn.HybridBlock):
    """
    Memory palace network for image classification
    dataset_size: size of data set: (N, C, H, W)
    cls_num: number of class
    cls_data_num: list of number of data in each class
    batch_size: batch_size of training data
    ctx: context
    """
    def __init__(self, batch_size, img_size, ctx, **kargs):
        super(MemoPalaNet, self).__init__()
        self.ctx = ctx
        self.resnet = ResNet()

        self.pala_size = [0, 0, 0]
        self.pala_size[0] = self.resnet.channels[-1]
        import math
        self.pala_size[1] = math.ceil(img_size[1] / self.resnet.decay_factor)
        self.pala_size[2] = math.ceil(img_size[2] / self.resnet.decay_factor)

        self.pool_size = 3
        self.batch_size = batch_size
        self.avg_pool_3D = nn.AvgPool3D(pool_size=self.pool_size, strides=1, padding=1)
        self.pos_kernel = self.get_pos_kernel(self.pala_size)
        self.kernel = self.pos_kernel.expand_dims(axis=0)
        self.eps = 1e-8



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

        sum_x = nd.sum(x, axis=[1, 2, 3])
        max_energy = nd.max(y, (1, 2, 3)) * self.pool_size ** 3 / (sum_x + self.eps)

        x2 = nd.expand_dims(x, axis=4)
        center_pos = nd.sum(x2 * self.kernel, axis=[1, 2, 3]) / (nd.expand_dims(sum_x, axis=1) + self.eps)

        return center_pos, max_energy

    def hybrid_forward(self, F, x):
        x = self.resnet(x)
        center_pos, max_energy = self.get_data_info(x)
        return center_pos, max_energy

"""
def my_loss(out, nc=10, na=5):
    eps = 1e-8
    data_center, max_energy = out
    overall_center = nd.mean(data_center, axis=0)
    cls_data_center = nd.reshape(data_center, (nc, na, 3))
    cls_center = nd.mean(cls_data_center, axis=1)
    intra_dis = nd.sum((cls_data_center - cls_center.expand_dims(axis=1)) ** 2)
    inter_dis = nd.sum((data_center - overall_center) ** 2)

    loss1 = nd.sum(nd.LeakyReLU(1 - max_energy - 0.2, slope=0.01, act_type='leaky') + 0.2)

    loss2 = - inter_dis / (intra_dis + eps)

    return(loss1, loss2)
"""
def my_loss(out, nc, ns, nq):
    eps = 1e-8
    data, max_energy = out
    data = data.astype('float64')
    max_energy = max_energy.astype('float64')

    cls_data = nd.reshape(data[0:nc*ns], (nc, ns, -1))
    cls_center = nd.mean(cls_data, axis=1)+1e-10
    data_center_dis = nd.norm(data[nc*ns:].expand_dims(axis=1) - cls_center.expand_dims(axis=0),
                              axis=2) ** 2
    weight = nd.zeros((nc*nq, nc), ctx=data.context, dtype='float64')
    for i in range(0, nc):
        weight[i*nq:i*nq+nq, i] = 1
    weight2 = 1 - weight
    loss1 = nd.sum(data_center_dis * weight)
    temp = nd.exp(- data_center_dis) * weight2
    loss2 = nd.max(nd.log(nd.sum(temp, axis=1) + 1e-40))

    loss3 = nd.sum(nd.LeakyReLU(1 - max_energy - 0.2, slope=0.01, act_type='leaky') + 0.2)

    label = nd.argmin(data_center_dis, axis=1)

    return (loss1 + loss2) / (nc * nq) + 0 * loss3, label, loss3

def cal_acc(label, nc, nq):
    correct_cnt = 0
    for i in range(nc):
        correct_cnt += nd.sum(label[i*nq:(i+1)*nq] == i).asscalar()
    return correct_cnt / (nc * nq)

import sys
if __name__ == '__main__':

    ctx = mx.gpu(1)
    nc = 50
    ns = 2
    nq = 2
    epoch_num = 400
    root_dir = '../omniglot/data'
    batch_size = nc * (ns + nq)

    train_loader, test_loader = Function.get_episode_lodaer(root_dir, (28, 28), nc, ns, nq)

    net = MemoPalaNet(batch_size, (3, 28, 28), ctx)
    test_data = nd.ones((batch_size, 3, 28, 28), ctx=ctx)
    net.initialize(init=init.Xavier(), ctx=ctx)

    lr_scheduler = mx.lr_scheduler.FactorScheduler(2000, 0.001, 0.5)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler':lr_scheduler})

    train_losses = []; train_accs = []
    test_losses = []; test_accs = []
    for epoch in range(1, epoch_num+1):
        train_loss = 0
        train_loss3 = 0
        train_acc = 0
        for data, cls_id, img_id in train_loader:
            data = data.copyto(ctx)

            with autograd.record():
                out = net(data)
                loss, label, loss3 = my_loss(out, nc, ns, nq)
            loss.backward()
            # print(net(data)[0], loss)
            trainer.step(batch_size * len(train_loader))
            train_loss += loss.asscalar()
            train_loss3 += loss3.asscalar()
            train_acc += cal_acc(label, nc, nq)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_loss3 /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if test_loader is not None:
            test_loss = 0
            test_acc = 0
            for data, cls_id, img_id in test_loader:
                data = data.copyto(ctx)
                out = net(data)
                loss, label, _ = my_loss(out, nc, ns, nq)

                test_loss += loss.asscalar()
                test_acc += cal_acc(label, nc, nq)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        print('epoch: %d train_loss3 %.4f train_acc: %.4f test_loss %.4f test_acc %.4f' %
              (epoch, train_loss3, train_acc, test_loss, test_acc))

        if epoch % 10 == 0:
            net.save_parameters(('../model/'+'mpn'+'_%04d' % (epoch)))
        pickle.dump(train_losses, open('../model/mpn_train_loss.pkl', 'wb'))
