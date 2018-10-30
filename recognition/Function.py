import os
import random
import math
import cv2
import mxnet as mx
import mxnet.ndarray as nd
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
    Loader images from a root_dir, it consists of some subdirs,
    whose names are labels of classes and contents are the corresponding image data.
    The names of image data should be intergers and are just the ids of them.
    """
    img_map = {}
    reverse_map = {}
    img_list = []
    cls_map = {}
    cls_reverse_map = {}
    cnt = 1
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
    Return a Dataset with item: (img, class_id, img_id), given a root dir
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
        return img_nd, class_id, img_id

    def __len__(self):
        return len(self.img_list)

def get_class_loader(img_root, img_list, img_map, img_size, batch_size=1):
    dataset = ClassDataset(img_root, img_list, img_map, img_size)
    return DataLoader(dataset, batch_size=batch_size)

class EpisodeLoader:
    def __init__(self, cls_loader, nc, ns, nq):
        cls_num = len(cls_loader)
        self.cls_num = cls_num

        temp = list(range(1, cls_num+1))
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
        img_ids_support = []
        img_ids_query = []
        cnt = 0
        while cnt < self.nc:
            try:
                cur_cls = next(self.cls_seq)
            except StopIteration:
                temp = list(range(1, self.cls_num+1))
                random.shuffle(temp)
                self.cls_seq = iter(temp)
                cur_cls = next(self.cls_seq)
            try:
                img, cls_id, img_id = next(self.ites[cur_cls])
            except StopIteration:
                self.ites[cur_cls] = iter(self.cls_loader[cur_cls])
                continue
            imgs_support.append(img[0:self.ns])
            imgs_query.append(img[self.ns:])
            cls_ids_support.append(cls_id[0:self.ns])
            cls_ids_query.append(cls_id[self.ns:])
            img_ids_support.append(img_id[0:self.ns])
            img_ids_query.append(img_id[self.ns:])

            cnt += 1
        self.cls_cnt += self.nc
        support = (nd.concatenate(imgs_support, 0), nd.concatenate(cls_ids_support, 0),
                   nd.concatenate(img_ids_support, 0))
        query = (nd.concatenate(imgs_query, 0), nd.concatenate(cls_ids_query, 0),
                 nd.concatenate(img_ids_query, 0))
        return nd.concatenate([support[0], query[0]], 0),\
                nd.concatenate([support[1], query[1]], 0),\
                nd.concatenate([support[2], query[2]], 0)

    def __len__(self):
        return math.ceil(self.cls_num / self.nc)

def get_episode_lodaer(img_root, img_size = (120, 100), nc=10, ns=5, nq=5, num_workers=0,
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
            dataset = ClassDataset(img_root, reverse_map[i][train_num:], img_map,
                                            img_size)
            data_loader[i] = DataLoader(dataset, batch_size=ns+nq, num_workers=0,
                                        last_batch='rollover', shuffle=True)
        test_loader = EpisodeLoader(data_loader, nc, ns, nq)

    return train_loader, test_loader

def plot_cls_data(anchor_data, test_data):
    start = 0.0
    stop = 1.0
    num_colors = len(anchor_data) + 1
    cm_sec = np.linspace(start, stop, num_colors)
    colors = [cm.jet(x) for x in cm_sec]
    pca_transform = PCA(n_components=3)

    temp = list(anchor_data.values())
    if test_data is not None:
        temp += list(test_data.values())
    all_data = nd.concatenate(temp, 0)
    pca_transform.fit(all_data.asnumpy())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for key in anchor_data.keys():
        cur_data = pca_transform.transform(anchor_data[key].asnumpy())
        ax.scatter(cur_data[:, 0], cur_data[:, 1], cur_data[:, 2], c=colors[key])
    plt.show()
