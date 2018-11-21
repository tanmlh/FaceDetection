import os
import random
import math
import cv2
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.gluon.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA


def get_ctx():
    """ get the current context """
    return mx.gpu()

def show_img(img):
    """
    Show an image or images of numpy or ndarray type
    """
    if type(img) is np.ndarray:
        cv2.imshow('a', img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
    else:
        if len(img.shape) == 3:
            img = nd.swapaxes(img, 0, 1)
            img = nd.swapaxes(img, 1, 2)
            img = img.asnumpy()
            cv2.imshow('a', img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
        else:
            for i in range(img.shape[0]):
                cur_img = img[i]
                cur_img = nd.swapaxes(cur_img, 0, 1)
                cur_img = nd.swapaxes(cur_img, 1, 2)
                cur_img = cur_img.asnumpy()
                cv2.imshow('a', cur_img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    cv2.destroyAllWindows()

def stick_boxes(img_np, boxes):
    crops = []
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :4]
        pt1_x = int(bbox[0])
        pt1_y = int(bbox[1])
        pt2_x = int(bbox[2])
        pt2_y = int(bbox[3])
        bbox_width = pt2_x - pt1_x
        bbox_height = pt2_y - pt1_y
        pad_width = int(np.round(bbox_width * 0.1))
        pad_height = int(np.round(bbox_height * 0.1))
        pt1_x = max(0, pt1_x - pad_width)
        pt1_y = max(0, pt1_y - pad_height)
        pt2_x = min(img_np.shape[1], pt2_x + pad_width)
        pt2_y = min(img_np.shape[0], pt2_y + pad_height)

        cv2.rectangle(img_np, (pt1_x, pt1_y), (pt2_x, pt2_y),\
                      (0, 255, 0), 2)

        crops.append(img_np[pt1_y:pt2_y, pt1_x:pt2_x])
    return img_np, crops

def chg_ctx(arg_params, aux_params, ctx):
    """change the context of dict-like parameters"""
    new_args = dict()
    new_auxs = dict()
    for key, value in arg_params.items():
        new_args[key] = value.as_in_context(ctx)
    for key, value in aux_params.items():
        new_auxs[key] = value.as_in_context(ctx)
    return new_args, new_auxs

def np2nd(img_np, ctx=get_ctx()):
    img_nd = nd.array(img_np, ctx=ctx)
    img_nd = nd.swapaxes(img_nd, 1, 2)
    img_nd = nd.swapaxes(img_nd, 0, 1)
    img_nd = nd.expand_dims(img_nd, 0)
    return img_nd

def nd2np(img_nd):
    img_np = img_nd.asnumpy()[0]
    img_np = img_np.swapaxes(0, 1)
    img_np = img_np.swapaxes(1, 2)
    return img_np

def resize_pad(img_np, des_size):
    ratio_src = img_np.shape[0] / img_np.shape[1]
    ratio_des = des_size[0] / des_size[1]
    if ratio_src > ratio_des:
        scale = des_size[0] / img_np.shape[0]
    else:
        scale = des_size[1] / img_np.shape[1]
    img_np = cv2.resize(img_np, None, None, fx=scale, fy=scale,\
                        interpolation=cv2.INTER_LINEAR)
    if ratio_src > ratio_des:
        delta = des_size[1]-img_np.shape[1]
        pad = (0, 0, delta//2, delta-delta//2)
    else:
        delta = des_size[0]-img_np.shape[0]
        pad = (delta//2, delta-delta//2, 0, 0)
    img_np = cv2.copyMakeBorder(img_np, pad[0], pad[1],\
                                pad[2], pad[3],\
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_np

def get_img_map(root_dir):
    """
    loader images from a root_dir, it consists of some subdirs,
    whose names are labels of classes and contents are the corresponding image data.
    the names of image data should be intergers and are just the ids of them.
    """
    img_map = {}
    reverse_map = {}
    img_list = []
    cls_map = {}
    cls_reverse_map = {}
    cnt = 0
    for sub_dir in os.listdir(root_dir):

        if cls_map.get(sub_dir) is None:
            cls_map[sub_dir] = cnt
            cls_reverse_map[cnt] = sub_dir
            cnt += 1

        for img_name in os.listdir(os.path.join(root_dir, sub_dir)):
            img_path = os.path.join(sub_dir, img_name)
            img_list.append(img_path)
            img_map[img_path] = cls_map[sub_dir]
            if cls_map[sub_dir] not in reverse_map.keys():
                reverse_map[cls_map[sub_dir]] = []
            reverse_map[cls_map[sub_dir]].append(img_path)

    return img_map, reverse_map, img_list, cls_map, cls_reverse_map

class ClassDataset:
    """
    return a dataset with item: (img, class_id, img_id), given a root dir
    """
    def __init__(self, img_root, img_list, img_map, img_size):
        self.img_root = img_root
        self.img_list = img_list
        self.img_map = img_map
        self.img_size = img_size

    def __getitem__(self, idx):
        img_np = cv2.imread(os.path.join(self.img_root, self.img_list[idx]))
        img_np = resize_pad(img_np, self.img_size)
        img_nd = nd.array(img_np)
        img_nd = nd.swapaxes(img_nd, 1, 2)
        img_nd = nd.swapaxes(img_nd, 0, 1)
        temp = self.img_list[idx]
        class_id = int(self.img_map[temp])
        img_id = idx
        return img_nd, class_id

    def __len__(self):
        return len(self.img_list)

def get_class_loader(img_root, img_list, img_map, img_size, batch_size=1):
    random.shuffle(img_list)
    dataset = ClassDataset(img_root, img_list, img_map, img_size)
    return DataLoader(dataset, batch_size=batch_size)

class EpisodeLoader:
    def __init__(self, cls_loader, nc, ns, nq):
        cls_num = len(cls_loader)
        self.cls_num = cls_num

        temp = list(cls_loader.keys())
        random.shuffle(temp)
        self.cls_seq = iter(temp)

        self.ites = {}
        for key in cls_loader.keys():
            self.ites[int(key)] = iter(cls_loader[key])

        self.cls_loader = cls_loader
        self.nc = nc
        self.ns = ns
        self.nq = nq
        self.cls_cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cls_cnt >= self.cls_num:
            self.cls_cnt = 0
            raise StopIteration
        imgs_support = []
        imgs_query = []
        cls_ids_support = []
        cls_ids_query = []
        # img_ids_support = []
        # img_ids_query = []
        cnt = 0
        while cnt < self.nc:
            try:
                cur_cls = next(self.cls_seq)
            except StopIteration:
                temp = list(self.cls_loader.keys())
                random.shuffle(temp)
                self.cls_seq = iter(temp)
                cur_cls = next(self.cls_seq)
            try:
                img, cls_id = next(self.ites[cur_cls])
            except StopIteration:
                self.ites[cur_cls] = iter(self.cls_loader[cur_cls])
                continue
            imgs_support.append(img[0:self.ns])
            imgs_query.append(img[self.ns:])
            cls_ids_support.append(cls_id[0:self.ns])
            cls_ids_query.append(cls_id[self.ns:])
            # img_ids_support.append(img_id[0:self.ns])
            # img_ids_query.append(img_id[self.ns:])

            cnt += 1
        self.cls_cnt += self.nc
        support = (nd.concatenate(imgs_support, 0), nd.concatenate(cls_ids_support, 0))
        query = (nd.concatenate(imgs_query, 0), nd.concatenate(cls_ids_query, 0))
        return nd.concatenate([support[0], query[0]], 0),\
                nd.concatenate([support[1], query[1]], 0)

    def __len__(self):
        return math.ceil(self.cls_num / self.nc)

def get_episode_lodaer(img_root, reverse_map, img_map, img_size, nc, ns, nq, num_workers=0):
    data_loader = {}
    for key in reverse_map.keys():
        dataset = ClassDataset(img_root, reverse_map[key], img_map, img_size)
        data_loader[key] = DataLoader(dataset, batch_size=ns+nq, num_workers=num_workers,
                                      last_batch='rollover', shuffle=True)
    loader = EpisodeLoader(data_loader, nc, ns, nq)
    return loader

def get_episode_lodaer_v2(img_root, img_size = (120, 100), nc=10, ns=5, nq=5, num_workers=0,
                       split=True):
    img_map, reverse_map, img_list, cls_map, _ = get_img_map(img_root)
    cls_num = len(reverse_map)
    cls_seq = list(range(1, cls_num+1))
    random.shuffle(cls_seq)
    data_loader = {}
    for i in range(1, cls_num+1):
        sample_num = len(reverse_map[i])
        if split:
            train_num = min(sample_num * 4 // 5, sample_num-1)
        else:
            train_num = sample_num
        dataset = ClassDataset(img_root, reverse_map[i][0:train_num], img_map,
                                        img_size)
        data_loader[i] = DataLoader(dataset, batch_size=ns+nq, num_workers=0,
                                    last_batch='rollover', shuffle=True)
    train_loader = EpisodeLoader(data_loader, nc, ns, nq)

    test_loader = []
    if split:
        data_loader = {}
        for i in range(1, cls_num+1):
            sample_num = len(reverse_map[i])
            train_num = min(sample_num * 4 // 5, sample_num-1)
            dataset = classdataset(img_root, reverse_map[i][train_num:], img_map,
                                            img_size)
            data_loader[i] = dataloader(dataset, batch_size=ns+nq, num_workers=0,
                                        last_batch='rollover', shuffle=true)
        test_loader = episodeloader(data_loader, nc, ns, nq)

    return train_loader, test_loader

def train_net(net, train_loader, test_loader, loss_fun, acc_fun, ctx, net_name,
              epoch_num=100, net_dir='../../model'):
    """
    train a network.
    """
    lr_scheduler = mx.lr_scheduler.FactorScheduler(2000, 0.001, 0.5)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler':lr_scheduler})
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(1, epoch_num+1):
        train_loss = 0
        train_acc = 0
        cnt = 0
        for data, label in train_loader:
            data = data.copyto(ctx)
            label = label.copyto(ctx)
            batch_size = data.shape[0]
            with autograd.record():
                out = net(data)
                loss = loss_fun(out, label)

            loss.backward()
            trainer.step(batch_size * len(train_loader))
            train_loss += loss.asscalar()
            train_acc += acc_fun(out, label)
            cnt += 1

        train_loss /= cnt
        train_acc /= cnt
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if test_loader is not None:
            test_loss = 0
            test_acc = 0
            cnt = 0
            for data, label in test_loader:
                data = data.copyto(ctx)
                label = label.copyto(ctx)
                out = net(data)
                loss = loss_fun(out, label)

                test_loss += loss.asscalar()
                test_acc += acc_fun(out, label)
                cnt += 1

            test_loss /= cnt
            test_acc /= cnt
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        if test_loader is None:
            print('epoch: %d train_loss: %.4f train_acc: %.4f' %
                  (epoch, train_loss, train_acc))

        else:
            print('epoch: %d train_loss: %.4f train_acc: %.4f test_loss %.4f test_acc %.4f' %
                  (epoch, train_loss, train_acc, test_loss, test_acc))

        if epoch % 50 == 0:
            net_path = os.path.join(net_dir, net_name)
            if not os.path.exists(net_path):
                os.mkdir(net_path)
            net.save_parameters(net_path + '/' + net_name + '_%04d' % (epoch))

    return net

class Recognizer:
    def __init__(self, model_prefix, epoch, ctx, img_size):
        sym, arg_params, auxparams = mx.model.load_checkpoint(model_prefix, epoch)
        all_layers = sym.get_internals()
        sym = sym[0]
        model = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
        model.bind(data_shapes=[('data', (1, 3, img_size[0], img_size[1]))])
        model.set_params(arg_params, auxparams)
        self.model = model
        self.ctx = ctx
        self.img_size = img_size

    def load_train_data(self, train_data, label, label_map=None):
        train_data = train_data.as_in_context(self.ctx)

        cls_center = {}
        cls_data_cnt = {}
        for i in range(train_data.shape[0]):
            cur_cls = label[i]
            if cls_center.get(cur_cls) is None:
                cls_center[cur_cls] = 0
                cls_data_cnt[cur_cls] = 0

            data_batch = mx.io.DataBatch(data=(train_data[i:i+1],))
            self.model.forward(data_batch, is_train=False)
            embedding = self.model.get_outputs()[0]
            embedding = nd.L2Normalization(embedding, mode='instance')
            cls_center[cur_cls] += embedding
            cls_data_cnt[cur_cls] += 1
        for key in cls_center:
            cls_center[key] /= cls_data_cnt[key]
        self.cls_center = cls_center
        self.label_map = label_map

    def predict(self, img):
        img_np = resize_pad(img, self.img_size)
        img_nd = np2nd(img_np, self.ctx).as_in_context(self.ctx)
        data_batch = mx.io.DataBatch(data=(img_nd,))
        self.model.forward(data_batch, is_train=False)
        embedding = self.model.get_outputs()[0]
        embedding = nd.L2Normalization(embedding, mode='instance')
        max_sim = -np.inf
        label = -1
        for key in self.cls_center:
            cur_dis = nd.sum(self.cls_center[key] * embedding)
            if cur_dis > max_sim:
                max_sim = cur_dis
                label = key
        if self.label_map is not None:
            label = self.label_map[label]

        return label, embedding



def plot_cls_data(anchor_data, test_data, dim=2):
    """
    plot the anchor data and test data in two or three dimensional space
    anchor_data: map of ndarray
    test_data: map of ndarray
    """
    start = 0.0
    stop = 1.0
    num_colors = len(anchor_data) + 1
    cm_sec = np.linspace(start, stop, num_colors)
    colors = [cm.jet(x) for x in cm_sec]

    if dim == 3:
        pca_transform = PCA(n_components=3)
        temp = list(anchor_data.values())
        if test_data is not None:
            temp += list(test_data.values())
        all_data = nd.concatenate(temp, 0)
        pca_transform.fit(all_data.asnumpy())

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        i = 0
        for key in anchor_data.keys():
            cur_data = pca_transform.transform(anchor_data[key].asnumpy())
            # cur_data = cur_data / np.sqrt(np.sum(cur_data**2))
            ax.scatter(cur_data[:, 0], cur_data[:, 1], cur_data[:, 2], c=colors[i], s=40)
            i += 1

        i = 0
        if test_data is not None:
            for key in test_data.keys():
                cur_data = pca_transform.transform(test_data[key].asnumpy())
                # cur_data = cur_data / np.sqrt(np.sum(cur_data**2))
                ax.scatter(cur_data[:, 0], cur_data[:, 1], cur_data[:, 2], c=colors[i], marker='s')
                i += 1
        plt.show()
    else:
        pca_transform = PCA(n_components=2)
        temp = list(anchor_data.values())
        if test_data is not None:
            temp += list(test_data.values())
        all_data = nd.concatenate(temp, 0)
        pca_transform.fit(all_data.asnumpy())
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        i = 0
        for key in anchor_data.keys():
            cur_data = pca_transform.transform(anchor_data[key].asnumpy())
            plt.scatter(cur_data[:, 0], cur_data[:, 1], c=colors[i], s=40)
            i += 1
        i = 0
        if test_data is not None:
            for key in test_data.keys():
                cur_data = pca_transform.transform(test_data[key].asnumpy())
                plt.scatter(cur_data[:, 0], cur_data[:, 1], c=colors[i], marker='s')
                i+= 1
        plt.show()

def plot_episode(data, nc, ns, nq, dim=3):
    anchor_data = {}
    test_data = {}
    for i in range(nc):
        anchor_data[i] = data[i*ns:(i+1)*ns]
    for i in range(nc):
        test_data[i] = data[nc*ns + i*nq : nc*ns + (i+1)*nq]
    plot_cls_data(anchor_data, test_data, dim)
