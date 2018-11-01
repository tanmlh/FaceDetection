"""Some function for face recognition"""
import pickle
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import init, gluon, autograd
from mxnet.gluon import nn
import resnet
from CelebA import get_data_loader
import cv2

def get_ctx():
    """get the current context"""
    return mx.gpu(1)
class SiameseNet(nn.HybridBlock):
    """
    Construct a siamese network
    """
    def __init__(self, ctx, **kwargs):
        super(SiameseNet, self).__init__(**kwargs)
        self.res_net1 = resnet.resnet18_v1(ctx=ctx)
        self.res_net2 = resnet.resnet18_v1(ctx=ctx, params=self.res_net1.collect_params())
        self.soft_max = nn.Dense(units=2)
        self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x, y):
        x = self.res_net1(x)
        y = self.res_net2(y)
        res = self.soft_max(self.flatten(nd.abs(x-y)))
        return res, x

def train_siamese_net():
    net = SiameseNet(ctx=get_ctx())
    net.initialize(init=init.Xavier(), ctx=get_ctx())
    batch_size = 4
    train_data_loader = get_data_loader(batch_size=batch_size, num_workers=2)
    test_data_loader = get_data_loader(batch_size=batch_size, num_workers=2)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.005})
    cnt = 0
    train_losses = []; test_losses = []
    train_accs = []; test_accs = []
    ephoch_size = 100
    for epoch in range(0, 100):
        train_loss = 0; test_loss = 0
        train_acc = 0; test_acc = 0

        for data1, data2, label in train_data_loader:
            data1 = data1.copyto(get_ctx())
            data2 = data2.copyto(get_ctx())
            label = label.copyto(get_ctx())

            with autograd.record():
                out, _ = net(data1, data2)
                loss = softmax_cross_entropy(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += acc(out, label)
            #if cnt % 100 == 0:
            #    print('progress: %.4f' % (cnt / len(train_data_loader) / ephoch_size * 100))
            cnt += 1

        for data1, data2, label in test_data_loader:
            data1 = data1.copyto(get_ctx())
            data2 = data2.copyto(get_ctx())
            label = label.copyto(get_ctx())
            out, _ = net(data1, data2)
            loss = softmax_cross_entropy(out, label)
            test_loss += loss.mean().asscalar()
            test_acc += acc(out, label)

        train_loss /= len(train_data_loader)
        test_loss /= len(test_data_loader)
        train_acc /= len(train_data_loader)
        test_acc /= len(test_data_loader)

        print('epoch: %d train_acc: %.4f train_loss: %.4f test_acc: %.4f' % \
              (epoch, train_acc, train_loss, test_acc))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if epoch % 10 == 0:
            net.save_parameters(('../model/res18x2_%04d' % (epoch)))

        pickle.dump(train_accs, open('../model/train_accs.pkl', 'wb'))
        pickle.dump(test_accs, open('../model/test_accs.pkl', 'wb'))

class ClassNet(nn.HybridBlock):
    def __init__(self):
        model_path = '../model/res18x2_0009'
        model = SiameseNet()
        model.load_parameters(model_path, ctx=get_ctx())
        self.model = model
        self.dense = nn.Dense()

def acc(output, label):
    """calculate the accurate"""
    return (output.argmax(axis=1) == label.T[0]).mean().asscalar()

def show_img_nd(img_nd):
    """use cv2.imshow to show an image with ndarray type"""
    img_nd = img_nd.swapaxes(0, 1)
    img_nd = img_nd.swapaxes(1, 2)
    img_np = img_nd.asnumpy()
    cv2.imshow('a', img_np)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    siamese_net = SiameseNet(get_ctx())
    siamese_net.load_parameters('../model/res18x2_0010', ctx=get_ctx())
