from detection.symbol import faster_rcnn
from detection.detection import detect_face
import mxnet as mx
import cv2
import os
import mxnet.ndarray as nd
import numpy as np
from mxnet.io import DataIter, DataBatch
from mxnet.gluon.data import DataLoader, Dataset
from mxnet.gluon.data.vision import transforms
import random

def get_ctx():
    return mx.cpu()

"""
def get_img_map(img_path, prefix=None):
    img_map = {}
    reverse_map = {}
    img_list = []
    img_path = '../crops'
    for dirpath, dirnames, filenames in os.walk(img_path):
        print(dirpath, dirnames, filenames)
        break
    return

    text_path = '../CelebA/Anno/identity_CelebA.txt'
    with open(text_path) as file:
        for line in file:
            line_split = line.split()
            img_name = os.path.join(line_split[1], line_split[0])
            if os.path.exists(os.path.join(img_path, img_name)):
                img_list.append(img_name)
                img_map[img_name] = line_split[1]
                if line_split[1] not in reverse_map.keys():
                    reverse_map[line_split[1]] = []
                reverse_map[line_split[1]].append(img_name)
    return img_map, reverse_map, img_list
"""
def get_img_map(root_dir='../crops'):
    img_map = {}
    reverse_map = {}
    img_list = []
    for sub_dir in os.listdir(root_dir):
        for img_name in os.listdir(os.path.join(root_dir, sub_dir)):
            img_path = os.path.join(sub_dir, img_name)
            img_list.append(img_path)
            img_map[img_path] = sub_dir
            if sub_dir not in reverse_map.keys():
                reverse_map[sub_dir] = []
            reverse_map[sub_dir].append(img_path)
    return img_map, reverse_map, img_list

def show_img(img_np):
    cv2.imshow('a', img_np)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

def get_img_iter(img_dir= \
                 '../CelebA/Img/img_celeba'):
    for file_name in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file_name)
        img_np = cv2.imread(file_path)
        show_img(img_np)

def get_data_iter(img_dir='../crops'):
    list_dir = os.listdir(img_dir)
    file_num = len(list_dir)
    data = nd.zeros((file_num, 6, 120, 100))
    for file_name in list_dir:
        file_path = os.path.join(img_dir, file_name)
        img_np = cv2.imread(file_path)
        img_nd = nd.array(img_np, ctx=get_ctx())

def chg_ctx(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for key, value in arg_params.items():
        new_args[key] = value.as_in_context(ctx)
    for key, value in aux_params.items():
        new_auxs[key] = value.as_in_context(ctx)
    return new_args, new_auxs

def resize_long(img_np, long_size):
    short_edge = min(img_np.shape[0:2])
    long_edge = max(img_np.shape[0:2])
    resize_scale = long_size / long_edge
    img_np = cv2.resize(img_np, None, None, fx=resize_scale,\
                    fy=resize_scale, interpolation=cv2.INTER_LINEAR)
    #return img_np, resize_scale
    delta = max(img_np.shape[0:2]) - min(img_np.shape[0:2])
    pad = (int(delta/2), delta-int(delta/2), 0, 0)
    if img_np.shape[0] > img_np.shape[1]:
        pad = (0, 0, int(delta/2), delta-int(delta/2))
    img_np = cv2.copyMakeBorder(img_np, pad[0], pad[1], pad[2], pad[3],\
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_np, resize_scale
def trans_img(img_np):
    img_nd = nd.array(img_np, ctx=get_ctx())
    img_nd = nd.swapaxes(img_nd, 1, 2)
    img_nd = nd.swapaxes(img_nd, 0, 1)
    img_nd = nd.expand_dims(img_nd, 0)
    return img_nd

def stick_boxes(img_np, boxes):
    crops = []
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :4]
        pt1_x = int(bbox[0])
        pt1_y = int(bbox[1])
        pt2_x = int(bbox[2])
        pt2_y = int(bbox[3])
        crops.append(img_np[pt1_y:pt2_y, pt1_x:pt2_x])
    return img_np, crops

def crop_faces():
    model_name = 'mxnet-face-fr50'
    _, arg_params, aux_params = mx.model.\
            load_checkpoint(model_name, 0)
    arg_params, aux_params = chg_ctx(arg_params, aux_params, get_ctx())
    sym = faster_rcnn.faster_rcnn50(num_class=2)
    img_dir = '../CelebA/Img/img_celeba'
    cnt = 0
    num_multi = 0
    num_fail = 0
    img_map = get_img_map()
    for file_name in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file_name)
        img_np = cv2.imread(file_path)
        img_np, scale = resize_long(img_np, 600)
        img_nd = trans_img(img_np)
        boxes = detect_face(get_ctx(), sym, \
                            arg_params, aux_params, img_nd)
        img_np, crops = stick_boxes(img_np, boxes)
        if len(crops) > 1:
            num_multi += 1
        elif len(crops) == 0:
            num_fail += 1
        if len(crops) != 0:
            cur_id = img_map[file_name]
            path = '../crops/'+ cur_id
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite("../crops/"+cur_id+'/'+file_name, \
                        crops[0])
        cnt += 1
        if cnt % 500 == 0:
            print('Already processed: %d' % cnt)
            print('Current multis: %d' % num_multi)
            print('Current fails: %d\n' % num_fail)
        #break
        #if cnt >= 20:
        #   break
def part_perm(num, num_ori):
    times = 1
    num_part = num_ori
    if num_ori > num:
        times = num_ori // num + 1
        num_part = num

    num_list = range(0, num)
    seg = np.linspace(0, num, num_part+1, dtype='int64')
    idx_lay1 = np.random.permutation(num_part)
    res = []
    for idx in idx_lay1:
        val = np.random.randint(seg[idx], \
                          seg[idx+1])
        res.append(val)
    temp_res = res
    for i in range(times-1):
        res.extend(temp_res)
    return res[0:num_ori]

def decode_random(random_code):
    random_code += 1
    l = 2; r = random_code+1
    while l <= r:
        mid = (l+r) // 2
        if mid*(mid-1)//2 >= random_code:
            r = mid-1
        else:
            l = mid+1
    p1 = r
    p2 = random_code - r*(r-1)//2 - 1
    return p1, p2

def random_pair(the_list, num = 1, cls_map = {}):
    size_list = len(the_list)
    pairs = []
    per = part_perm((size_list*(size_list-1))//2, num*2)
    cnt = 0
    for random_code in per:
        p1, p2 = decode_random(random_code)
        if cls_map == {}:
            pairs.append((the_list[p1], the_list[p2], 0))
        else:
            if cls_map[the_list[p1]] == cls_map[the_list[p2]]:
                continue
            else:
                pairs.append((the_list[p1], the_list[p2], 1))
        cnt += 1
        if cnt >= num:
            break
    return pairs

def get_pair_num_list(cls_size_list, pair_num):
    sum_size = 0
    num_cls = 0
    seg = [0]
    for i in cls_size_list:
        if i == 1:
            continue
        num_cls += 1
        sum_size += i
        seg.append(sum_size)
    scale_f = lambda x : round(x*pair_num/sum_size)
    scale_seg = list(map(scale_f, seg))
    pair_num_list = []
    for i in range(num_cls):
        pair_num_list.append(scale_seg[i+1] - scale_seg[i])
    return pair_num_list

def get_img_pair_list():
    img_map, reverse_map, img_list = get_img_map()
    num_pairs_pos = len(img_list) // 2
    num_pairs_neg = len(img_list) // 2

    cls_size_list = []
    for key in reverse_map.keys():
        cls_size_list.append(len(reverse_map[key]))
    pair_num_list = get_pair_num_list(cls_size_list, num_pairs_pos)
    cnt = 0
    pair_list_pos = []
    for key in reverse_map.keys():
        cls_size = len(reverse_map[key])
        cls_size_list.append(cls_size)
        if cls_size <= 1:
            continue
        pair_list_pos.extend(random_pair(reverse_map[key],\
                                     pair_num_list[cnt]))
        cnt += 1
    pair_list_neg = random_pair(img_list, num_pairs_neg, img_map)
    return pair_list_pos, pair_list_neg

"""
class PairImgIter(DataIter):
    def __init__(self, batch_size, img_pair_list,\
                 img_root='../crops', data_name='data', \
                 label_name = 'label', img_size=(120, 100)):
        super(PairImgIter, self).__init__()
        self.batch_size = batch_size
        self.img_root = img_root
        self.img_pair_list = img_pair_list
        self.data_size = len(img_pair_list)
        self.data_name = data_name
        self.label_name = label_name
        self.img_size = img_size
        self.cursor = 0

    @property
    def provide_data(self):
        return [self.data_name, (self.batch_size, 3,\
                                 self.img_size[0],\
                                 self.img_size[1])]
    @property
    def provide_label(self):
        return [self.label_name, (self.batch_size, 1)]

    def resize_pad(self, img_np):
        des_size = self.img_size
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

    def _read_img(self, img_name, ):
        img_np = cv2.imread(os.path.join(self.img_root, img_name))
        img_np = self.resize_pad(img_np)
        img_nd = nd.array(img_np, ctx=get_ctx())
        img_nd = nd.swapaxes(img_nd, 1, 2)
        img_nd = nd.swapaxes(img_nd, 0, 1)
        img_nd = nd.expand_dims(img_nd, 0)
        return img_nd

    def next(self):
        if self.cursor < self.data_size:
            if self.cursor + self.batch_size > self.data_size:
                self.cursor = self.data_size - self.batch_size
            data1 = nd.zeros(self.provide_data[1], ctx=get_ctx())
            data2 = nd.zeros(self.provide_data[1], ctx=get_ctx())
            label = nd.zeros(self.provide_label[1], ctx=get_ctx())
            _cursor = self.cursor
            cnt = 0
            for i in range(_cursor, _cursor + self.batch_size):
                img1 = self._read_img(self.img_pair_list[i][0])
                img2 = self._read_img(self.img_pair_list[i][1])
                data1[cnt] = img1[0]
                data2[cnt] = img2[0]
                label[cnt] = int(self.img_pair_list[i][2])
                cnt+=1
                self.cursor += 1
            return DataBatch([data1, data2], [label])
        else:
            raise StopIteration
"""
class PairImgDataset(Dataset):
    def __init__(self, pair_img_list, img_root='../crops', \
                 img_size=(120, 100)):
        super(PairImgDataset, self)
        self.pair_img_list = pair_img_list
        self.img_root = img_root
        self.img_size = img_size

    def __len__(self):
        return len(self.pair_img_list)

    def _resize_pad(self, img_np):
        des_size = self.img_size
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

    def _read_img(self, img_name):
        img_np = cv2.imread(os.path.join(self.img_root, img_name))
        img_np = self._resize_pad(img_np)
        img_nd = nd.array(img_np, ctx=get_ctx())
        img_nd = nd.swapaxes(img_nd, 1, 2)
        img_nd = nd.swapaxes(img_nd, 0, 1)
        trans = transforms.Normalize(0.13, 0.31)
        img_nd = trans(img_nd)
        return img_nd

    def __getitem__(self, idx):
        data1 = self._read_img(self.pair_img_list[idx][0])
        data2 = self._read_img(self.pair_img_list[idx][1])
        cls = self.pair_img_list[idx][2]
        label = nd.array([cls], ctx=get_ctx())
        return (data1, data2, label)

def get_data_loader(batch_size=64, num_workers=8):
    pair_list_pos, pair_list_neg = get_img_pair_list()
    pair_list = pair_list_pos
    pair_list.extend(pair_list_neg)
    random.shuffle(pair_list)
    pair_img_dataset = PairImgDataset(pair_list)
    pair_img_dataloader = DataLoader(pair_img_dataset, batch_size=64, \
                                     num_workers = num_workers)
    return pair_img_dataloader

if __name__ == '__main__':
    pass
