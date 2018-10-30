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

def rename_folder(root_dir='../OptFace/facetrain_less'):
    cnt = 1
    for sub_dir in os.listdir(root_dir):
        for img_name in os.listdir(os.path.join(root_dir, sub_dir)):
            img_path = os.path.join(root_dir, os.path.join(sub_dir, img_name))
            os.rename(img_path, os.path.join(root_dir, os.path.join(sub_dir, str(cnt)+'.jpg')))
            cnt += 1

def test_proto_net():
    nc = 10
    ns = 3
    nq = 3
    img_size = (120, 100)
    train_img_root = '../OptFace/facetrain_less'
    test_img_root = '../OptFace/face_test'
    _, _, train_img_list, train_cls_map, train_reverse_cls_map = Function.get_img_map(train_img_root)
    _, _, test_img_list, test_cls_map, test_reverse_cls_map = Function.get_img_map(test_img_root)

    train_loader, test_loader = Function.get_episode_lodaer(train_img_root, nc, ns, nq)
    net = ProtoNet.ResNet()
    # net.load_parameters('../model/protoNet_0090', mx.gpu(1))

    # ProtoNet.train_proto_net(1, train_loader, test_loader, 'Proto_net_tune', net, nc, ns, nq)
    net.load_parameters('../model/Proto_net_tune_0090')

    train_dataset = Function.ClassDataset(train_img_root, train_img_list, train_cls_map, img_size)
    test_dataset = Function.ClassDataset(test_img_root, test_img_list, test_cls_map, img_size)
    train_loader = DataLoader(train_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    ProtoNet.attach_labels(net, train_loader)
    for data, cls_id, img_id in test_loader:
        cls_id = cls_id.asscalar()
        print(ProtoNet.predict(net, data), cls_id)

def train(gpu_id, model, net, train_loader, test_loader, net_name, nc=10, ns=3, nq=3, epoch_num=100):
    ctx = mx.gpu(gpu_id)
    batch_size = nc * (ns + nq)

    # train_loader, test_loader = CelebA.get_episode_lodaer(nc, ns, nq)

    lr_scheduler = mx.lr_scheduler.FactorScheduler(2000, 0.001, 0.5)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler':lr_scheduler})
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(1, epoch_num+1):
        train_loss = 0
        train_acc = 0
        for data, cls_id, img_id in train_loader:
            data = data.copyto(ctx)
            model.forward(data, is_train=False)
            data = model.get_outputs()

            with autograd.record():
                out = net(data)
                loss, label = my_loss(out, nc, ns, nq)

            loss.backward()
            # print(net(data)[0], loss)
            trainer.step(batch_size * len(train_loader))
            train_loss += loss.asscalar()
            train_acc += cal_acc(label, nc, nq)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if test_loader is not None:
            test_loss = 0
            test_acc = 0
            for data, cls_id, img_id in test_loader:
                data = data.copyto(ctx)
                out = net(data)
                loss, label = my_loss(out, nc, ns, nq)

                test_loss += loss.asscalar()
                test_acc += cal_acc(label, nc, nq)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        print('epoch: %d train_loss: %.4f train_acc: %.4f test_loss %.4f test_acc %.4f' %
              (epoch, train_loss, train_acc, test_loss, test_acc))

        if epoch % 50 == 0:
            if not os.path.exists('../model/' + net_name):
                os.mkdir('../model/' + net_name)
            net.save_parameters('../model/' + net_name + '/' + net_name + '_%04d' % (epoch))

    return net

class MyDataset:
    def __init__(self, data, cls_id, img_id):
        self.data = data
        self.cls_id = cls_id
        self.img_id = img_id

    def __getitem__(self, idx):
        return (self.data[idx], self.cls_id[idx], idx)

    def __len__(self):
        return self.data.shape[0]

def get_embedding_loader(data_loader, model, batch_size=32, ctx=mx.gpu(1), nc=10, ns=3, nq=3):
    all_data = {}
    all_cls_id = {}
    all_img_id = {}
    label = nd.ones((batch_size,))
    for data, cls_id, img_id in data_loader:
        data = data.as_in_context(ctx)

        data_batch = mx.io.DataBatch(data=(data,))
        model.forward(data_batch, is_train=False)
        data2 = model.get_outputs()[0]
        data2 = data2.asnumpy()
        data2 = nd.array(data2, ctx=ctx)
        for i in range(data.shape[0]):
            cur_cls_id = cls_id[i].asscalar()
            if all_data.get(cur_cls_id) == None:
                all_data[cur_cls_id] = []
                all_cls_id[cur_cls_id] = []
                all_img_id[cur_cls_id] = []
            all_data[cur_cls_id].append(data2)
            all_cls_id[cur_cls_id].append(cls_id)
            all_img_id[cur_cls_id].append(img_id)

    loader = {}
    for key in all_data.keys():
        data = nd.concatenate(all_data[key], 0)
        cls_id = nd.concatenate(all_cls_id[key], 0)
        img_id = nd.concatenate(all_img_id[key], 0)
        loader[key] = DataLoader(MyDataset(data, cls_id, img_id),
                                 batch_size=ns+nq, shuffle=True, last_batch='rollover')

    return Function.EpisodeLoader(loader, nc, ns, nq)

if __name__ == '__main__':
    nc = 50
    ns = 10
    nq = 10
    img_size = (120, 100)
    batch_size = 32
    img_size = [112, 112]
    gpu_id = 1
    ctx = mx.gpu(gpu_id)
    train_img_root = '../OptFace/facetrain_less'
    test_img_root = '../OptFace/face_test'

    model_name = '../model/model-r50-am-lfw/model'

    train_img_map, _, train_img_list, _, _ = Function.get_img_map(train_img_root)
    test_img_map, _, test_img_list, _, _ = Function.get_img_map(test_img_root)

    # train_loader, _ = Function.get_episode_lodaer(train_img_root, img_size, nc, ns, nq, split=False)
    train_loader = Function.get_class_loader(train_img_root, train_img_list, train_img_map, img_size)
    test_loader = Function.get_class_loader(test_img_root, test_img_list, test_img_map, img_size)



    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
    #arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (batch_size, 3, img_size[0], img_size[1]))])
    model.set_params(arg_params, aux_params)

    train_loader = get_embedding_loader(train_loader, model, batch_size, ctx, nc, ns, nq)
    test_loader = get_embedding_loader(test_loader, model, batch_size, ctx, nc, ns, nq)

    net = nn.Sequential()
    # net.add(nn.BatchNorm)
    net.add(nn.Dense(512))
    # net.load_parameters('../model/resnet50_tuned/resnet50_tuned_0800', ctx=ctx)
    net.initialize(init.Xavier(), ctx=ctx)
    ProtoNet.train_proto_net(gpu_id, train_loader, test_loader, 'resnet50_tuned', net, nc, ns, nq,
                             epoch_num=1000)


    cls_data = {}
    temp, _, _ = next(iter(train_loader))
    for i in range(10):
        cls_data[i] = net(temp[i*ns:i*ns+5].as_in_context(ctx))
    Function.plot_cls_data(cls_data, None)

