"""
Some function about extracting the face-involved frames of a video
"""
import os
import sys
import queue
import random
import mxnet as mx
import cv2
sys.path.append('../')
from detection.detection import Detector
from common import Function
import fcntl
from tracking import sort
from recognition import OptFace


def dump_frames(video_writer, frames):
    """
    Dump the frames in a queue into the files through a video_writer
    """
    while not frames.empty():
        frame = frames.get()
        video_writer.write(frame)

def compress_video(video_dir, des_dir, ctx_id=0, video_id=0):
    """
    Extracted the face-involved frames and output to des_dir
    """
    ctx = mx.gpu(ctx_id)
    video_capture = cv2.VideoCapture(video_dir)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter.fourcc('D', 'I', 'V', 'X')
    video_writer = cv2.VideoWriter(des_dir, fourcc, int(fps), size)
    video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frame = 0

    model_name = '../detection/mxnet-face-fr50'
    _, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
    arg_params, aux_params = Function.chg_ctx(arg_params, aux_params, ctx)
    sym = faster_rcnn.faster_rcnn50(num_class=2)

    buffer_size = 50
    frame_buffer = queue.Queue(buffer_size)
    record_mode = False
    missed_frames_cnt = 0
    while video_capture.grab():
        flag, frame = video_capture.retrieve()
        frame_nd = F.np2nd(frame, ctx=ctx)

        if not flag:
            continue
        num_frame += 1
        if num_frame % 500 == 0:
            print('Video id: %d  Num frame: %d\n' % (video_id, num_frame))
        # put the current frame into a buffer
        frame_buffer.put(frame)

        boxes = detect_face(ctx, sym, arg_params, aux_params, frame_nd)
        if len(boxes) > 0:
            if not record_mode:
                record_mode = True
                dump_frames(video_writer, frame_buffer)
            elif frame_buffer.full():
                dump_frames(video_writer, frame_buffer)
        elif record_mode:
            missed_frames_cnt += 1
            if frame_buffer.full():
                dump_frames(video_writer, frame_buffer)

            if missed_frames_cnt > buffer_size:
                dump_frames(video_writer, frame_buffer)
                record_mode = False
                missed_frames_cnt = 0
        if frame_buffer.full():
            frame_buffer.get()

def get_video_list():
    video_dir = '../OptFace/Video'
    video_list = []
    for root, dirs, files in os.walk(video_dir):
        for a_file in files:
            file_name = os.path.join(root, a_file)
            video_list.append(file_name)
    with open('../OptFace/video_list.txt', 'w') as f:
        for file_name in video_list:
            if file_name.endswith('.mp4'):
                f.write(file_name + ' 0\n')

def compress_video_list(ctx_id):
    compressed_video_dir = '../OptFace/CompressedVideo'
    video_list_dir = '../OptFace/video_list.txt'

    while True:
        video_file = open(video_list_dir, 'r+')
        fcntl.flock(video_file, fcntl.LOCK_EX)

        lines = video_file.readlines()
        cur_video_id = int(lines[0])
        if cur_video_id >= len(lines)-1:
            video_file.close()
            break

        lines[0] = str(cur_video_id+1)
        cur_video_name = lines[cur_video_id+1][:-3]
        video_file.seek(0)
        video_file.write(lines[0]+'\n')
        video_file.close()

        compress_video(cur_video_name, os.path.join(compressed_video_dir, \
                                                    'compressed_video_'+lines[0]+'.avi'), \
                       ctx_id, cur_video_id)

def dump_crop(root_dir, crop, cls_id, img_id):
    file_dir = os.path.join(root_dir, str(cls_id))
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    cv2.imwrite(os.path.join(file_dir, str(img_id)+'.png'), crop)

def extract_tracked_frames():
    gpu_id = 1
    ctx = mx.gpu(gpu_id)
    video_path = '../../dataset/OptFace/CompressedVideo'
    detector = Detector('../../model/faster_rcnn/mxnet-face-fr50', ctx)

    img_id_map = {}
    img_cnt = 0
    cls_cnt = 0
    for video_file in os.listdir(video_path):
        frames_dir = os.path.join('../../dataset/OptFace/tracked_frames', video_file)
        if not os.path.exists(frames_dir):
            os.mkdir(frames_dir)

        print('processing video: %s' % video_file)
        video_dir = os.path.join(video_path, video_file)
        video_capture = cv2.VideoCapture(video_dir)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        num_frame = 0
        tracker = sort.Sort()
        while video_capture.grab():
            flag, frame = video_capture.retrieve()
            det = detector.detect_face(frame)
            ids = tracker.update(det)
            # if ids.shape[0] > 0:
            #     print(ids)
            sticked_img, crops = Function.stick_boxes(frame, ids)

            for i in range(ids.shape[0]):
                cur_id = int(ids[i][4])
                if img_id_map.get(cur_id) is None:
                    img_id_map[cur_id] = cls_cnt
                    cls_cnt += 1
                dump_crop(frames_dir, crops[i], img_id_map[cur_id], img_cnt)
                img_cnt += 1

            """
            cv2.imshow('a', sticked_img)
            if cv2.waitKey(5) == ord('q'):
                break
            """
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


def split_train_test():
    root_dir = '../../dataset/OptFace/CompressedVideo'
    videos = os.listdir(root_dir)
    random.shuffle(videos)
    train_num = len(videos) * 4 // 5
    train_videos = videos[:train_num]
    test_videos = videos[train_num:]
    train_file = open('../../dataset/OptFace/train_videos.txt', 'w')
    test_file = open('../../dataset/OptFace/test_videos.txt', 'w')

    for line in train_videos:
        train_file.writelines(line + '\n')

    for line in test_videos:
        test_file.writelines(line + '\n')

def get_frames_info(file_list='../../dataset/OptFace/train_videos.txt'):
    frames_dir = '../../dataset/OptFace/tracked_frames'
    train_video_list = [l.strip() for l in open(file_list)]
    img_map = {}
    reverse_map = {}
    cls_map = {}
    cls_reverse_map = {}
    img_list = []
    min_len = 10
    cls_cnt = 0


    for video_name in train_video_list:
        path1 = os.path.join(frames_dir, video_name)

        for track_id in os.listdir(path1):
            path2 = os.path.join(path1, track_id)

            if len(os.listdir(path2)) > min_len:

                cls_map[int(track_id)] = cls_cnt
                reverse_map[cls_cnt] = []
                cls_reverse_map[cls_cnt] = int(track_id)

                for img_id in os.listdir(path2):
                    img_path = os.path.join(os.path.join(video_name, track_id), img_id)
                    img_list.append(img_path)
                    img_map[img_path] = cls_cnt
                    reverse_map[cls_cnt].append(img_path)
                cls_cnt += 1

    return img_list, img_map, reverse_map, cls_map, cls_reverse_map


def get_episode_loader(img_size, nc, ns, nq):
    frames_dir = '../../dataset/OptFace/tracked_frames'
    _, img_map, reverse_map, _, _ = get_frames_info()


    episode_loader = Function.get_episode_lodaer(frames_dir, reverse_map, img_map,
                                                 img_size, nc, ns, nq)
    return episode_loader

if __name__ == '__main__':
    gpu_id = 1
    ctx = mx.gpu(gpu_id)
    video_path = '../../dataset/OptFace/CompressedVideo'
    detector = Detector('../../model/faster_rcnn/mxnet-face-fr50', ctx)
    video_list = '../../dataset/OptFace/test_videos.txt'

    img_size = (112, 112)
    model_prefix = '../../model/arcface_tune_level1/arcface_tune_level1'
    model_prefix = '../../model/model-r50-am-lfw/model'
    recognizer = OptFace.Recognizer(model_prefix, 0, ctx, img_size)
    train_data, label, reverse_cls_map = OptFace.get_train_data()
    recognizer.load_train_data(train_data, label, reverse_cls_map)

    img_id_map = {}
    img_cnt = 0
    cls_cnt = 0

    video_list = [l.strip() for l in open(video_list)]
    for video_file in video_list[0:1]:

        print('processing video: %s' % video_file)
        video_dir = os.path.join(video_path, video_file)
        video_capture = cv2.VideoCapture(video_dir)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        num_frame = 0
        tracker = sort.Sort()
        while video_capture.grab():
            flag, frame = video_capture.retrieve()
            det = detector.detect_face(frame)
            ids = tracker.update(det)
            sticked_img, crops = Function.stick_boxes(frame, ids)
            for crop in crops:
                crop_name = recognizer.predict(crop)
                print('crop name: %s' % (crop_name))

            # sys.exit()
            cv2.imshow('a', sticked_img)
            if cv2.waitKey(5) == ord('q'):
                break
