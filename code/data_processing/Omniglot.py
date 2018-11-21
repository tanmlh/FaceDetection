import os
import shutil
import numpy as np
from mxnet import autograd, init
import mxnet as mx
import Function
import ProtoNet
import cv2

def rename_folder(root_dir='../omniglot/data'):
    cnt = 1
    root_dir2 = '../omniglot/data2'
    for sub_dir in os.listdir(root_dir):
        path1 = os.path.join(root_dir, sub_dir)
        for sub_dir2 in os.listdir(path1):
            path2 = os.path.join(path1, sub_dir2)
            for img_name in os.listdir(path2):
                img_path = os.path.join(path2, img_name)
                img_np = cv2.imread(img_path)
                folder_path = os.path.join(root_dir2, sub_dir + '_' + sub_dir2[-2:])
                for direction in range(4):
                    img_np = np.rot90(img_np)
                    temp = folder_path+'_'+str(direction)
                    if not os.path.exists(temp):
                        os.mkdir(temp)
                    cv2.imwrite(os.path.join(temp, str(cnt)+'.png'), img_np)
                    cnt += 1

def test_one_shot(root_dir, net, ctx):
    avg_acc = 0
    for running in os.listdir(root_dir):
        cur_dir = os.path.join(root_dir, running)
        f = open(os.path.join(cur_dir, 'class_labels.txt'), 'r')
        cls_id = 1
        img_map = {}
        train_img_list = []
        test_img_list = []
        for line in f.readlines():
            line = line.strip('\n')
            test_file, train_file = line.split(' ')
            img_map[train_file] = cls_id
            img_map[test_file] = cls_id
            train_img_list.append(train_file)
            test_img_list.append(test_file)
            cls_id += 1
        train_loader = Function.get_class_loader(root_dir, train_img_list, img_map, (28, 28))
        test_loader = Function.get_class_loader(root_dir, test_img_list, img_map, (28, 28))
        ProtoNet.attach_labels(net, train_loader, ctx)
        label, acc = ProtoNet.predict(net, test_loader, ctx)
        avg_acc += acc
        print('%s: %4f' % (running, acc))
    print('avg: %4f' % (avg_acc / 20))


if __name__ == '__main__':
    # rename_folder()
    # import sys
    # sys.exit()
    ctx_id = 1
    ctx=mx.gpu(ctx_id)

    root_dir = '../omniglot/data'
    img_map, reverse_map, img_list, cls_map, cls_reverse_map = Function.get_img_map(root_dir)
    nc = 60
    ns = 5
    nq = 5
    train_loader, test_loader = Function.get_episode_lodaer(root_dir, (28, 28), nc, ns, nq)
    net = ProtoNet.ResNet(thumbnail=True)
    net.load_parameters('../model/ProtoNet_omniglot2_0200', ctx=ctx)
    # net.initialize(init=init.Xavier(), ctx=ctx)
    # net = ProtoNet.train_proto_net(ctx_id, train_loader, test_loader, 'ProtoNet_omniglot2',
    #                                nc=nc, ns=ns, nq=nq, epoch_num=1000)
    test_one_shot('../omniglot/one_shot', net, ctx)
