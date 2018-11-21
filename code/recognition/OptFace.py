import os
import math
import cv2
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader, Dataset
import sys
sys.path.append('../')
import ProtoNet
from common import Function
from common import Loss
from data_processing import ProcessVideo


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

def get_model(model_path, img_size, batch_size, ctx):
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


def train_arcface_softmax():
# if __name__ == '__main__':
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

    model = get_model(model_path, img_size, batch_size, ctx)

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
    label_list = [0, 1, 2, 3, 4, 6]
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

def get_tuned_sym(embedding, pre_embedding, nc, ns, nq, lam=1, margin=0.4):
    embedding = mx.symbol.L2Normalization(data=embedding, mode='instance')
    pre_embedding = mx.symbol.L2Normalization(data=pre_embedding, mode='instance')

    s_embedding = embedding.slice_axis(axis=0, begin=0, end=nc*ns)
    q_embedding = embedding.slice_axis(axis=0, begin=nc*ns, end=None)

    s_cls_data = mx.symbol.reshape(s_embedding, (nc, ns, -1))
    q_cls_data = mx.symbol.reshape(q_embedding, (nc, nq, -1))

    s_cls_center = mx.symbol.mean(s_cls_data, axis=1)
    s_cls_center = mx.symbol.L2Normalization(s_cls_center, mode='instance')
    s_center_broadcast = s_cls_center.expand_dims(axis=1)
    s_center_dis = mx.symbol.sum(mx.symbol.broadcast_mul(q_cls_data, s_center_broadcast),
                                 axis=2)
    temp = mx.symbol.LeakyReLU(margin - s_center_dis, act_type='leaky', slope=0.1)
    loss1 = mx.symbol.sum(temp)

    """
    s_cls_center = mx.symbol.mean(s_cls_data, axis=1)
    s_center_broadcast = s_cls_center.expand_dims(axis=1).broadcast_axes(axis=1, size=nq)
    temp1 = mx.symbol.norm(q_cls_data - s_center_broadcast, axis=1)
    loss1 = mx.symbol.sum(temp1) / (nc * ns)
    """

    temp2 = mx.symbol.norm(embedding - pre_embedding, axis=1)
    loss2 = mx.symbol.sum(temp2)
    loss = mx.symbol.broadcast_add(loss1, lam * loss2)
    loss = mx.symbol.make_loss(loss, name='my_loss')

    return mx.symbol.Group([mx.sym.BlockGrad(embedding), mx.sym.BlockGrad(s_center_dis), loss])

def load_data(img_root):
    img_size = (112, 112)
    train_img_map, _, train_img_list, cls_map, reverse_cls_map = Function.get_img_map(img_root)
    train_data = []
    label = []
    for file_name in train_img_list:
        file_path = os.path.join(img_root, file_name)
        img_np = cv2.imread(file_path)
        img_np = Function.resize_pad(img_np, img_size)
        img_nd = Function.np2nd(img_np)
        train_data.append(img_nd)
        label.append(train_img_map[file_name])
    train_data = nd.concatenate(train_data, axis=0)

    return train_data, label, reverse_cls_map

def evaluate_model(model_prefix, epoch_num):

    train_img_root = '../../dataset/OptFace/facetrain_less'
    test_img_root = '../../dataset/OptFace/face_test3'
    train_data, train_label, train_cls_map = load_data(train_img_root)
    test_data, test_label, test_cls_map = load_data(test_img_root)

    recognizer = Function.Recognizer(model_prefix, epoch_num, ctx, img_size)
    recognizer.load_train_data(train_data, train_label, train_cls_map)
    acc = 0

    for i in range(test_data.shape[0]):
        pred_label, embedding = recognizer.predict(Function.nd2np(test_data[i:i+1]))
        if pred_label == test_cls_map[test_label[i]]:
            acc += 1
        # print(pred_label, test_cls_map[test_label[i]])
    print(acc / test_data.shape[0])
    acc /= test_data.shape[0]
 
    label_set_num = 10
    label_set = []
    anchor_data_map = {}
    test_data_map = {}
    for value in train_cls_map.values():
        if value in test_cls_map.values():
            label_set.append(value)
            label_set_num -= 1
            if label_set_num == 0:
                break
    print(label_set)
    for i in range(train_data.shape[0]):
        cur_label = train_cls_map[train_label[i]]
        if cur_label in label_set:
            if anchor_data_map.get(cur_label) == None:
                anchor_data_map[cur_label] = []
            cur_data = Function.nd2np(train_data[i:i+1])
            anchor_data_map[cur_label].append(recognizer.predict(cur_data)[1].as_in_context(mx.cpu()))
    for key in anchor_data_map.keys():
        anchor_data_map[key] = nd.concatenate(anchor_data_map[key], axis=0)

    for i in range(test_data.shape[0]):
        cur_label = test_cls_map[test_label[i]]
        if cur_label in label_set:
            if test_data_map.get(cur_label) == None:
                test_data_map[cur_label] = []
            cur_data = Function.nd2np(test_data[i:i+1])
            test_data_map[cur_label].append(recognizer.predict(cur_data)[1].as_in_context(mx.cpu()))
    for key in test_data_map.keys():
        test_data_map[key] = nd.concatenate(test_data_map[key], axis=0)

    """
    test_data_map = {}
    for key in anchor_data_map.keys():
        cur_data = anchor_data_map[key]
        temp = nd.broadcast_div(cur_data, nd.norm(cur_data, axis=1).expand_dims(axis=1))
        w = nd.sum(temp, axis=0)
        w = w / nd.norm(w)
        test_data_map[key] = w.expand_dims(axis=0)
    """
    Function.plot_cls_data(anchor_data_map, test_data_map, 2)
    # Function.plot_cls_data(anchor_data_map, None, 3)

    return acc

if __name__ == '__main__':
    ctx_id = 0
    ctx_id2 = 1
    ctx = mx.gpu(ctx_id)
    ctx2 = mx.gpu(ctx_id2)
    model_path = '../../model/model-r50-am-lfw/model'
    model_path = '../../model/arcface_tune_level1/arcface_tune_level1'
    model_epoch = 20
    nc = 3; ns = 10; nq = 10; img_size = (112, 112)


    # model_prefix = '../../model/arcface_tune_level1/arcface_tune_level1'
    model_prefix = '../../model/model-r50-am-lfw/model'
    acc = evaluate_model(model_prefix, 0)
    # print(acc)
    sys.exit()

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
    arg_params, aux_params = Function.chg_ctx(arg_params, aux_params, ctx)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    pre_embedding = mx.symbol.Variable('pre_embedding_label')
    sym_tuned = get_tuned_sym(sym, pre_embedding, nc, ns, nq, lam=1)

    model_fixed = mx.mod.Module(symbol=sym,
                                context=ctx2,
                                data_names=['data'],
                                label_names=None)

    model_tuned = mx.mod.Module(symbol=sym_tuned,
                                context=ctx,
                                data_names=['data'],
                                label_names=['pre_embedding_label'])
    """
    model_tuned = mx.mod.Module(symbol=sym_tuned,
                                context=ctx,
                                data_names=['data'])
    """
    model_fixed.bind(data_shapes=[('data', (nc*(ns+nq), 3, img_size[0], img_size[1]))])
    model_tuned.bind(data_shapes=[('data', (nc*(ns+nq), 3, img_size[0], img_size[1]))],
                     label_shapes=[('pre_embedding_label', (nc*(ns+nq), 512))])
    # model_tuned.bind(data_shapes=[('data', (nc*(ns+nq), 3, img_size[0], img_size[1]))])

    # mx.viz.plot_network(sym_tuned).view()
    # sys.exit()

    model_fixed.set_params(arg_params, aux_params)
    model_tuned.set_params(arg_params, aux_params)

    model_tuned.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 0.05),))
    metric = Loss.AccMetric(nc, ns, nq)
    # metric=mx.metric.create(acc_metric)

    train_loader = ProcessVideo.get_episode_loader(img_size, nc, ns, nq)

    # training process
    accs = []
    accs2 = []
    losses = []
    for epoch in range(1, 101):
        acc = 0
        acc2 = 0
        loss = 0
        for data, cls_id in train_loader:
            data = data.as_in_context(ctx)
            cls_id = cls_id.as_in_context(ctx)
            data_batch = mx.io.DataBatch(data=(data.as_in_context(ctx2),))
            model_fixed.forward(data_batch, is_train=False)

            pre_embedding = model_fixed.get_outputs()[0].as_in_context(ctx)
            temp = metric.get(pre_embedding, None)
            acc2 += temp[0]

            data_batch = mx.io.DataBatch(data=(data,), label=(pre_embedding,))
            # data_batch = mx.io.DataBatch(data=(data,))
            model_tuned.forward(data_batch, is_train=True)
            post_embedding, s_center_dis, cur_loss = model_tuned.get_outputs()
            model_tuned.backward()
            model_tuned.update()
            temp = metric.get(post_embedding, None)
            acc += temp[0]
            loss += cur_loss.asscalar()


        acc /= len(train_loader)
        acc2 /= len(train_loader)
        loss /= len(train_loader)
        accs.append(acc)
        accs2.append(acc2)
        losses.append(loss)

        print('Epoch %d, acc2: %f, acc: %f loss: %f' % (epoch, acc2, acc, loss))
        arg_params, aux_params = model_tuned.get_params()
        if epoch % 5 == 0:
            mx.model.save_checkpoint('../../model/arcface_tune_level1/arcface_tune_level1',
                                     epoch, sym_tuned, arg_params, aux_params)

