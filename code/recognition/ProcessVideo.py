"""
Some function about extracting the face-involved frames of a video
"""
import os
import sys
import queue
import mxnet as mx
import cv2
sys.path.append('../')
from detection.detection import Detector
import Function
import fcntl


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

if __name__ == '__main__':
    gpu_id = 1
    ctx = mx.gpu(gpu_id)
    video_path = '../../dataset/OptFace/CompressedVideo'
    detector = Detector('../../model/faster_rcnn/mxnet-face-fr50', ctx)
    for video_file in os.listdir(video_path):
        video_dir = os.path.join(video_path, video_file)
        video_capture = cv2.VideoCapture(video_dir)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        num_frame = 0
        while video_capture.grab():
            flag, frame = video_capture.retrieve()
            det = detector.detect_face(frame)
            sticked_img, _ = Function.stick_boxes(frame, det)
            cv2.imshow('a', sticked_img)
            if cv2.waitKey(5) == ord('q'):
                break
            print(det)
        break

