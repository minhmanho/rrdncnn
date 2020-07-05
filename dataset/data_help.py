import cv2
import numpy as np
import os
import time

def yuv420_to_yuv444(yuv):
    total_h, w = yuv.shape
    h = int(total_h/1.5)
    uv_h = h//4
    _w = int(w/2)
    _h = int(h/2)
    y = yuv[0:h, :]

    u = cv2.resize(np.reshape(yuv[h:h+uv_h, :], (_h, _w)), (w, h), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(np.reshape(yuv[h+uv_h:, :], (_h, _w)),  (w, h), interpolation=cv2.INTER_NEAREST)
    return np.stack((y, u, v), axis=2)

class VideoProcessor:
    def __init__(self, video_url, h=None, w=None):
        assert (w is not None or h is not None), "Width and Height are required to read rawvideo"
        self.w = int(w)
        self.h = int(h)
        self.frame_size = self.w * self.h
        self.frame_length = int((self.frame_size * 3) / 2)
        self.video_size = os.stat(video_url)[6]
        self.num_frames = int(self.video_size / self.frame_length)
        print(video_url, self.w, self.h, self.frame_size, self.frame_length, self.num_frames)
        self.f = open(video_url, 'rb')
        self.vid_arr = []
        for i in range(self.num_frames):
            _, frame = self.read()
            self.vid_arr.append(frame)
        self.release()

    def read(self):
        try:
            raw = self.f.read(self.frame_length)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape((int(self.h*1.5), self.w))
            frame = yuv420_to_yuv444(yuv)
            return True, frame
        except:
            return False, None

    def release(self):
        self.f.close()

