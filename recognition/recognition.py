import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gpu, init, gluon, autograd
from mxnet.gluon import nn
import resnet
from CelebA import get_data_loader
import cv2

def get_ctx():
    return mx.gpu(1)

class SiameseNet(nn.HybridBlock):
    def __init__(self, ctx, **kwargs):
        super(SiameseNet, self).__init__(**kwargs)
        self.res_net1 = resnet.resnet18_v1(ctx=ctx)
        self.res_net2 = \
                resnet.resnet18_v1(ctx=ctx, \
                                   params=self.res_net1.collect_params())
        self.soft_max = nn.Dense(units=2)
        self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x, y):
        x = self.res_net1(x)
        y = self.res_net2(y)
        # z = nd.concat(x, y, dim=1)
        res = self.soft_max(self.flatten(nd.abs(x-y)))
        return res

def acc(output, label):
    return (output.argmax(axis=1) == label.T[0]).mean().asscalar()

def show_img_nd(img_nd):
    img_nd = img_nd.swapaxes(0, 1)
    img_nd = img_nd.swapaxes(1, 2)
    img_np = img_nd.asnumpy()
    cv2.imshow('a', img_np)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    net = SiameseNet(ctx=get_ctx())
    net.initialize(init=init.Xavier(), ctx=get_ctx())
    batch_size=16
    data_loader = get_data_loader(batch_size=16, num_workers=8)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', \
                            {'learning_rate':0.005})
    cnt = 0
    train_losses = []
    ephoch_size = 100
    for epoch in range(0, 100):
        train_loss = 0
        train_acc = 0
        for data1, data2, label in data_loader:
            data1 = data1.copyto(get_ctx())
            data2 = data2.copyto(get_ctx())
            label = label.copyto(get_ctx())

            with autograd.record():
                out = net(data1, data2)
                loss = softmax_cross_entropy(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += acc(out, label)
            if cnt % 1000 == 0:
                print('progress: %.4f' % (cnt / len(data_loader) / ephoch_size * 100))
            cnt += 1
        print('epoch: %d train_acc: %.4f train_loss: %.4f' % \
              (epoch, train_acc / len(data_loader), \
               train_loss / len(data_loader)))
        train_losses.append(train_loss)
        if epoch % 10 == 0:
            net.save_parameters(('../model/res18x2_%04d' % (epoch//10)))
