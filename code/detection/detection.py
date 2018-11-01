# coding: utf-8
import numpy as np
import cv2
import mxnet as mx
import argparse, time
from .symbol.faster_rcnn import *
from .symbol.config import config
from .symbol.processing import bbox_pred, clip_boxes, nms
import sys
sys.path.append('../')
from recognition import Function

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

class Detector:
    """
    Faster RCNN Detector, give model path and context to initialize it
    """
    def __init__(self, model_path, ctx, epoch=0):
        config.TEST.RPN_MIN_SIZE = 30
        _, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
        self.arg_params, self.aux_params = ch_dev(arg_params, aux_params, ctx)
        self.sym = faster_rcnn50(num_class=2)
        self.ctx = ctx

    def detect_face(self, img_np, nms_thresh=0.3, thresh=0.8):
        img_nd = Function.np2nd(img_np, self.ctx)
        self.arg_params["data"] = img_nd
        im_info = mx.ndarray.array([[img_nd.shape[2], img_nd.shape[3], 1]], ctx=self.ctx)
        self.arg_params["im_info"] = im_info
        exe = self.sym.bind(self.ctx, self.arg_params, args_grad=None, grad_req="null",
                            aux_states=self.aux_params)


        exe.forward(is_train=False)
        output_dict = {name: nd for name, nd in zip(self.sym.list_outputs(), exe.outputs)}
        rois = output_dict['rpn_rois_output'].asnumpy()[:, 1:]  # first column is index
        scores = output_dict['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output_dict['bbox_pred_reshape_output'].asnumpy()[0]
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, (img_nd.shape[2],\
                                     img_nd.shape[3]))
        cls_boxes = pred_boxes[:, 4:8]
        cls_scores = scores[:, 1]
        keep = np.where(cls_scores >= thresh)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets.astype(np.float32), nms_thresh)
        dets = dets[keep, :]
        dets[:, :4] = (dets[:, :4]).round().astype(np.int32)

        return dets


def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def init_detector(prefix='detection/mxnet-face-fr50', epoch=0, gpu=0):
    config.TEST.RPN_MIN_SIZE = 24
    ctx = mx.gpu(gpu)
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    sym = faster_rcnn50(num_class=2)

    return arg_params, aux_params, sym, ctx




def main():
    ctx = mx.gpu(args.gpu)
    _, arg_params, aux_params = mx.model.load_checkpoint('mxnet-face-fr50', 0)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    sym = faster_rcnn50(num_class=2)

    #cap = cv2.VideoCapture('/home/mit/zhangxq/disk2/人脸表情视频/00012.MTS')
    cap = cv2.VideoCapture('../face_data/demov1.mp4')
    face_region_id = 0;
    while (cap.isOpened()):
        # Capture frame-by-frame
        tic = time.time()
        ret, color = cap.read()
        if ret == False:
            break
        # color = cv2.imread(args.img)
        img = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        img, scale = resize(img.copy(), args.scale, args.max_scale)
        im_info = np.array([[img.shape[0],\
                             img.shape[1], scale]],\
                           dtype=np.float32)  # (h, w, scale)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # change to (c, h, w) order
        img = img[np.newaxis, :]  # extend to (n, c, h, w)

        arg_params["data"] = mx.nd.array(img, ctx)
        arg_params["im_info"] = mx.nd.array(im_info, ctx)
        exe = sym.bind(ctx, arg_params, args_grad=None, grad_req="null", \
                       aux_states=aux_params)

        # tic = time.time()
        exe.forward(is_train=False)
        # toc = time.time()
        output_dict = {name: nd for name, nd in zip(sym.list_outputs(), exe.outputs)}
        rois = output_dict['rpn_rois_output'].asnumpy()[:, 1:]  # first column is index
        scores = output_dict['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output_dict['bbox_pred_reshape_output'].asnumpy()[0]
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, (im_info[0][0], im_info[0][1]))
        cls_boxes = pred_boxes[:, 4:8]
        cls_scores = scores[:, 1]
        keep = np.where(cls_scores >= args.thresh)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets.astype(np.float32), args.nms_thresh)
        dets = dets[keep, :]
        # toc = time.time()

        # print "time cost is:{}s".format(toc - tic)
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            pt1_x = int(round(bbox[0]/scale))
            pt1_y = int(round(bbox[1]/scale))
            pt2_x = int(round(bbox[2]/scale))
            pt2_y = int(round(bbox[3]/scale))
            cv2.imwrite("extracted_faces/face_region_"\
                        + str(face_region_id) + ".png",\
                        color[pt1_y:pt2_y, pt1_x:pt2_x])
            cv2.rectangle(color, \
                          (int(round(bbox[0] / scale)),\
                                  int(round(bbox[1] / scale))),\
                          (int(round(bbox[2] / scale)),\
                           int(round(bbox[3] / scale))),\
                          (0, 255, 0), 2)
        face_region_id += 1;
        # Display the resulting frame
        cv2.imshow('face detection', color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        toc = time.time()
        print("time cost is:{}s".format(toc - tic))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use pre-trainned resnet model to classify one image")
    parser.add_argument('--img', type=str, default='test.jpg', help='input image for classification')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--prefix', type=str, default='mxnet-face-fr50', help='the prefix of the pre-trained model')
    parser.add_argument('--epoch', type=int, default=0, help='the epoch of the pre-trained model')
    parser.add_argument('--thresh', type=float, default=0.8,
                        help='the threshold of face score, set bigger will get more'
                             'likely face result')
    parser.add_argument('--nms-thresh', type=float, default=0.3, help='the threshold of nms')
    parser.add_argument('--min-size', type=int, default=24, help='the min size of object')
    parser.add_argument('--scale', type=int, default=600, help='the scale of shorter edge will be resize to')
    parser.add_argument('--max-scale', type=int, default=1000, help='the maximize scale after resize')
    args = parser.parse_args()
    config.END2END = 1
    config.TEST.HAS_RPN = True
    config.TEST.RPN_MIN_SIZE = args.min_size
    config.SCALES = (args.scale,)
    config.MAX_SIZE = args.max_scale
    main()
