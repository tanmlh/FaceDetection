import os
import math
import cv2
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader, Dataset
import ProtoNet
import Function

def rename_folder(root_dir='../../dataset/OptFace/train_less'):
    cnt = 1
    for sub_dir in os.listdir(root_dir):
        for img_name in os.listdir(os.path.join(root_dir, sub_dir)):
            img_path = os.path.join(root_dir, os.path.join(sub_dir, img_name))
            os.rename(img_path, os.path.join(root_dir, os.path.join(sub_dir, str(cnt)+'.jpg')))
            cnt += 1

class MyDataset:
    def __init__(self, data, cls_id):
        self.data = data
        self.cls_id = cls_id

    def __getitem__(self, idx):
        return (self.data[idx], self.cls_id[idx].asscalar())

    def __len__(self):
        return self.data.shape[0]

def get_embedding_loader(data_loader, model, is_episode=False, batch_size=32, 
                         ctx=mx.gpu(1)):
    """
    Generating a data loader containing the embedding of the data in the given data_loader,
    produced by model
    """
    all_data = {}
    all_cls_id = {}

    all_data2 = []
    all_cls_id2 = []
    for data, label in data_loader:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        data_batch = mx.io.DataBatch(data=(data,))
        model.forward(data_batch, is_train=False)
        data2 = model.get_outputs()[0]
        data2 = data2.asnumpy()
        data2 = nd.array(data2, ctx=ctx)

        all_data2.append(data2)
        all_cls_id2.append(label)

        for i in range(data.shape[0]):
            cur_cls_id = label[i].asscalar()
            if all_data.get(cur_cls_id) == None:
                all_data[cur_cls_id] = []
                all_cls_id[cur_cls_id] = []
            all_data[cur_cls_id].append(data2)
            all_cls_id[cur_cls_id].append(label)

    if is_episode:
        loader = {}
        for key in all_data.keys():
            data = nd.concatenate(all_data[key], 0)
            cls_id = nd.concatenate(all_cls_id[key], 0)
            loader[key] = DataLoader(MyDataset(data, cls_id),
                                     batch_size=batch_size, shuffle=True, last_batch='rollover')
    else:
        all_data2 = nd.concatenate(all_data2, 0)
        all_cls_id2 = nd.concatenate(all_cls_id2, 0)
        loader = DataLoader(MyDataset(all_data2, all_cls_id2), batch_size=batch_size, shuffle=True,
                                      last_batch='rollover')
    return loader

def get_model(model_path, img_size, batch_size):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (batch_size, 3, img_size[0], img_size[1]))])
    model.set_params(arg_params, aux_params)

    return model

def loss_fun(out, label):
    soft_max_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    return nd.sum(soft_max_loss(out, label))

def acc_fun(out, label):
    out_label = nd.argmax(out, axis=1)
    return nd.mean(out_label.astype('float32') == label.astype('float32')).asscalar()


if __name__ == '__main__':
    nc = 50
    ns = 10
    nq = 10
    img_size = (120, 100)
    batch_size = 32
    img_size = [112, 112]
    gpu_id = 1
    ctx = mx.gpu(gpu_id)
    train_img_root = '../../dataset/OptFace/facetrain_less'
    test_img_root = '../../dataset/OptFace/face_test'

    model_path = '../../model/model-r50-am-lfw/model'

    model = get_model(model_path, img_size, batch_size)

    train_img_map, _, train_img_list, cls_map, _ = Function.get_img_map(train_img_root)
    test_img_map, _, test_img_list, _, cls_reverse_map = Function.get_img_map(test_img_root)
    for key in test_img_map.keys():
        old_cls_id = test_img_map[key]
        test_img_map[key] = cls_map[cls_reverse_map[old_cls_id]]

    train_loader = Function.get_class_loader(train_img_root, train_img_list, train_img_map,
                                             img_size, batch_size)
    test_loader = Function.get_class_loader(test_img_root, test_img_list, test_img_map,
                                            img_size, batch_size)
    train_loader = get_embedding_loader(train_loader, model, False, batch_size, ctx)
    test_loader = get_embedding_loader(test_loader, model, False, batch_size, ctx)

    net = nn.Sequential()
    # net.add(nn.BatchNorm)
    net.add(nn.Dense(512, activation='tanh'))
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(128, activation='tanh'))
    net.add(nn.Dense(50))

    #net.initialize(init.Xavier(), ctx=ctx)
    #Function.train_net(net, train_loader, test_loader, loss_fun, acc_fun, ctx, 'arcface_softmax')

    net.load_parameters('../../model/arcface_softmax/arcface_softmax_0100', ctx=ctx)

    cls_data = {}
    label_list = [0, 1, 2, 3, 4, 5, 6]
    for data, label in train_loader:
        data = net(data.as_in_context(ctx))
        for i in range(data.shape[0]):
            label_scalar = label[i].asscalar()
            if label_scalar in label_list:
                if cls_data.get(label_scalar) == None:
                    cls_data[label_scalar] = []
                cls_data[label_scalar].append(data[i:i+1])

    cls_data2 = {}
    for data, label in test_loader:
        data = net(data.as_in_context(ctx))
        for i in range(data.shape[0]):
            label_scalar = label[i].asscalar()
            if label_scalar in label_list:
                if cls_data2.get(label_scalar) == None:
                    cls_data2[label_scalar] = []
                cls_data2[label_scalar].append(data[i:i+1])

    for key in cls_data.keys():
        cls_data[key] = nd.concatenate(cls_data[key], 0)
    for key in cls_data2.keys():
        cls_data2[key] = nd.concatenate(cls_data2[key], 0)
    """
    cls_data = {}
    temp, __ = next(iter(train_loader))
    for i in range(10):
        cls_data[i] = net(temp[i*ns:i*ns+5].as_in_context(ctx))
    """

    Function.plot_cls_data(cls_data, cls_data2)
