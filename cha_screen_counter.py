#!/user/bin/env python3 -tt
"""
Screen counting within GE's Centricity High Acuity (CHA).

Records timestamps and frames of screen changes from
screen capture video.

Created by Eric J. Robinson
"""

# Imports
import sys
import subprocess
import argparse
import os
import pdb
import numpy as np
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from IPython.display import display
import random
from skimage.metrics import structural_similarity as ssim
import time
import datetime


# Global variables

# Class declarations
class FrameCounter(object):
    """docstring for FrameCounter"""
    def __init__(self, args):
        self.args = args
        self.video = args.video
        self.base_name = self.video.split('/')[-1].split('.')[0]

        self.comparison_dir = os.path.join('data', self.base_name)
        self.paired_dir = os.path.join(self.comparison_dir, 'paired_frames')
        self.single_dir = os.path.join(self.comparison_dir, 'single_frames')
        self.dup_dir = os.path.join(self.comparison_dir,'duplicates')
        self.areas_dir = os.path.join(self.comparison_dir, 'areas_of_interest')
        self.omit_dir = os.path.join(self.comparison_dir, 'omit_dir')

        self.timestamps = []
        self.mse_list = []
                    
        self.cap = cv2.VideoCapture(self.video)
        while not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video)
            cv2.waitKey(1000)
            print("Wait for header")

        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))

        os.makedirs(self.comparison_dir, exist_ok=True)
        os.makedirs(self.paired_dir, exist_ok=True)
        os.makedirs(self.single_dir, exist_ok=True)
        os.makedirs(self.dup_dir, exist_ok=True)
        os.makedirs(self.areas_dir, exist_ok=True)
        os.makedirs(self.omit_dir, exist_ok=True)

        self.mse_threshold = args.threshold
        self.neighbor_threshold = 30

        self.scale_percent = 50

        self.freq = self.args.freq
        self.now = time.time()



        # [x1, x2, y1, y2]
        # 003 and 004
        aoi_1 = [0, 808, 108, 142]
        aoi_2 = [0, 1234, 986, 1038]
        default_width = 1620
        default_height = 1080

        # 005 (for some reason it's a different dimension?)
        # aoi_1 = [0, 730, 83, 113]
        # aoi_2 = [0, 863, 646, 687]

        # default_width = 1080
        # default_height = 720

        aois = [aoi_1, aoi_2]

        # aoi_1 = [0, 730, 49, 73]
        # aoi_2 = [275, 1920, 80, 120]
        # aoi_3 = [220, 1550, 151, 219]
        # aois = [aoi_1, aoi_2, aoi_3]

        self.areas_of_interest = []
        for aoi in aois:
            self.mse_list.append([])
            self.areas_of_interest.append(
                [aoi[0] * (self.frame_width / default_width),
                aoi[1] * (self.frame_width / default_width),
                aoi[2] * (self.frame_height / default_height),
                aoi[3] * (self.frame_height / default_height)])

        # self.key_coords = [
        # int(412 * (self.frame_width / default_width)),
        # int(416 * (self.frame_width / default_width)),
        # int(689 * (self.frame_height / default_height)),
        # int(697 * (self.frame_height / default_height))
        # ] 
        for i in range(0, len(self.areas_of_interest)):
            for j in range(0, len(self.areas_of_interest[i])):
                self.areas_of_interest[i][j] = int(self.areas_of_interest[i][j])

    def compare_frames(self):
        count = 0
        last = 0
        save_after_lag = 0
        lagging = False
        prev_aoi = []
        screen_change = []
        prev_frame = None
        skipping = False

        for area in self.areas_of_interest:
            prev_aoi.append(None)
            screen_change.append(None)

        while self.cap.isOpened():
            if count % 100 == 0:
                self.printProgressBar(count, self.frame_count)

            ret,frame = self.cap.read()
            i = 0
            if frame is None:
                break
            elif count >= 100 and not (count % self.freq):
                if save_after_lag > 0:
                    save_after_lag -= 1
                    if save_after_lag == 0:
                        # print(f'saving at frame {count}')
                        self.plot_images(lag_frame, frame, count, lag_err)
                else:
                    err = 0
                    for area in self.areas_of_interest:
                        x1, x2, y1, y2 = area

                        if prev_aoi[i] is None:
                            first_pass = True
                            prev_aoi[i] = frame[y1:y2,x1:x2,:]
                        aoi_1 = prev_aoi[i]
                        aoi_2 = frame[y1:y2,x1:x2,:]
                        # img = Image.fromarray(aoi, 'RGB')
                        # img.show()
                        if not first_pass:
                            screen_change[i], aoi_err = self.analyze_aoi(aoi_1, aoi_2)
                            if aoi_err > err:
                                err = aoi_err
                            self.log_mse_changes(i, aoi_err, count)
                            

                        prev_aoi[i] = aoi_2
                        first_pass = False
                        i += 1
                    
                    if self.change_exists(screen_change):
                        if self.not_neighbor(count, last):
                            if self.args.lag > 0:
                                save_after_lag = self.args.lag // self.freq
                                lag_count = count
                                lag_frame = prev_frame
                                lag_err = err
                                # print(f'lag set: count = {count}, err = {int(err)}')
                            else:
                                # print(f'saving frame {count} without lag')
                                self.plot_images(prev_frame, frame, count, err)
                            last = count
                        else:
                            self.plot_images(prev_frame, frame, count, err, path=self.dup_dir)
                            # print(f"skipping {count} for {last-1}\n")
                            last = count
                            # skipping = True
            
            prev_frame = frame
            count += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def log_mse_changes(self, idx, err, count):
        self.mse_list[idx].append((count, err))
        return

    def analyze_aoi(self, aoi_1, aoi_2):
        # Mean Standard Error
        err = np.sum((aoi_1 - aoi_2) ** 2)
        err /= float(aoi_1.shape[0] * aoi_1.shape[1])

        if err > self.mse_threshold:
            return True, err
        else:
            return False, err

    def plot_images(self, img1, img2, count, err, path=None):
        f, axarr = plt.subplots(2,1, figsize=(10,15))
        axarr[0].imshow(img1, aspect='auto')
        axarr[1].imshow(img2, aspect='auto')
        if path:
            plt.savefig(os.path.join(path,f'{count:06d}.jpg'))
        else:
            self.timestamps.append((count / self.frame_rate, count))
            self.save_aoi(img2, count)
            if err < self.mse_threshold:
                pdb.set_trace()
            plt.savefig(os.path.join(self.paired_dir,f'{count:06d}_e{int(err):03d}_paired.jpg'))
            cv2.imwrite(os.path.join(self.single_dir,f'{count:06d}.jpg'), img2)
        plt.close(f)


    def scale_down(self, img, pct):
        # Unused helper method
        width = int(img.shape[1] * pct / 100)
        height = int(img.shape[0] * pct / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    def change_exists(self, screen_changes):
        if self.args.ssim:
            return screen_changes
        else:
            val = False
            for change in screen_changes:
                val = val or change
            return val

    def not_neighbor(self, count, last):
        return (count - last) >= self.neighbor_threshold

    def save_aoi(self, img, count):
        start = True
        for area in self.areas_of_interest:
            x1, x2, y1, y2 = area

            new = img[y1:y2,x1:x2,:]

            if start:
                aois = img[y1:y2,x1:x2,:]
                start = False
            else:
                if new.shape[1] > aois.shape[1]:
                    aois = np.concatenate((aois, np.zeros((aois.shape[0], new.shape[1]-aois.shape[1], aois.shape[2]))), axis = 1)
                else:
                    new = np.concatenate((new, np.zeros((new.shape[0], aois.shape[1]-new.shape[1], new.shape[2]))), axis = 1)
                aois = np.concatenate((aois, new), axis=0)
        cv2.imwrite(os.path.join(self.areas_dir,'{:06d}.jpg'.format(count)), aois)

    def save_timestamps(self):
        with open(os.path.join(self.comparison_dir, 'timestamps.txt'), 'w') as f:
            writer = csv.writer(f)
            for time in self.timestamps:
                writer.writerow(["{}".format(time[0]), "{}".format(time[1])])

    def load_times(self):
        time_file = os.path.join(self.args.videos, self.base_name) + '.txt'
        gt = []
        with open(time_file) as f:
            for line in f:
                gt.append(int(float(line) * self.frame_rate))
        return gt

    def get_axis(self, ax):
        xmin, xmax, ymin, ymax = ax.axis()
        return ymin, ymax

    def plot_mse(self):
        dot_size = 10
        gt = self.load_times()
        gt = np.asarray(gt) // self.freq
        # fig, axs = plt.subplots(len(self.mse_list)+1, 2)
        fig, axs = plt.subplots(1, 2)
        idx = 0
        # if self.args.edge:
        #     edge_array = np.asarray(self.edge_sums, dtype=np.int32)
        #     edge_diff = np.abs(edge_array - np.roll(edge_array,1))
        #     edge_diff = edge_diff[1:]

        #     axs[idx, 0].scatter(np.arange(edge_array.shape[0]), edge_array, s=dot_size, c='darkorange')
        #     axs[idx, 0].title.set_text('Raw edge counts')
        #     axs[idx, 1].scatter(np.arange(edge_diff.shape[0]), edge_diff, s=dot_size, c='darkorange')
        #     axs[idx, 1].title.set_text('Frame-frame edge count changes')
        # elif self.args.ssim:
        #     ssim_array = np.asarray(self.ssims, dtype=np.float64)
        #     ssim_hist = ssim_array[ssim_array > 0.03]

        #     axs[0].scatter(np.arange(ssim_array.shape[0]), ssim_array, s=dot_size, c='darkorange')
        #     axs[0].title.set_text('Raw ssim counts')
        #     axs[1].hist(ssim_hist, facecolor='darkorange', bins=70)
        #     axs[1].title.set_text('(1 - SSIM) distribution')


        # ymin, ymax = self.get_axis(axs[idx, 0])
        # axs[idx, 0].vlines(gt, ymin, ymax, colors='cornflowerblue')
        axs[0].vlines(gt, 0, 1, colors='cornflowerblue')
        ymin, ymax = self.get_axis(axs[idx, 1])
        axs[idx, 1].vlines(gt, ymin, ymax, colors='cornflowerblue')
        
        axs[idx, 0].hist(edge_array, bins=100)
        axs[idx, 1].hist(edge_diff, bins=100)

        idx += 1
        # for aoi in self.mse_list:
        #     anp = np.asarray(aoi)
        #     color = "#%06x" % random.randint(0, 0xFFFFFF)
        #     x = anp[:,0]
        #     y = anp[:,1]
        #     mask = y > 5
        #     x = x[mask]
        #     y = y[mask]
        #     axs[idx, 0].scatter(x, y, c=color, s=dot_size)
        #     axs[idx, 0].title.set_text('AOI mean squared error')
        #     axs[idx, 1].hist(y, facecolor=color, bins=70)
        #     axs[idx, 1].title.set_text('AOI mse distribution')


        #     ymin, ymax = self.get_axis(axs[idx, 0])
        #     axs[idx, 0].vlines(gt, ymin, ymax, colors='cornflowerblue')
        #     # ymin, ymax = self.get_axis(axs[idx, 1])
        #     # axs[idx, 1].vlines(gt, ymin, ymax, colors='cornflowerblue')
        #     idx += 1
        # plt.show()

    def printProgressBar(self, count, total, prefix = '', suffix = '', decimals = 1, length = 40, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            count       - Required  : current count (Int)
            total       - Required  : total counts (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        if count == 0:
            return

        percent = ("{0:." + str(decimals) + "f}").format(100*(float(count) / float(total)))
        filledLength = int(length * count // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}%', end = printEnd)
        # Print New Line on Complete
        if count == total: 
            print()

# Function declarations
def determine_pickle(pickle_base, ptype):
    pickle = os.path.join(pickle_base, ptype + '.obj')
    if os.path.exists(pickle):
        return pickle
    else:
        return None

def load_pickles(args, pickle_base, fc):
    # pickled = determine_pickle(pickle_base, 'mse')
    pickled = os.path.join(pickle_base, 'mse.obj')
    if pickled and args.load:
        with open(pickled, 'rb') as f:
            print(f'loading pickle {pickled}...')
            fc.mse_list = pickle.load(f)

def write_pickles(args, pickle_base, fc):
    # pickled = determine_pickle(pickle_base, 'mse')
    pickled = os.path.join(pickle_base, 'mse.obj')
    if pickled:
        with open(pickled, 'wb') as f:
            print(f'writing pickle {pickled}...')
            pickle.dump(fc.mse_list, f)

def main(args):
    for _, _, files in os.walk(args.videos):
        for file in files:
            if file.endswith('mp4'):
                print('processing {}'.format(file))
                args.video = os.path.join(args.videos,file)
                pickle_base = os.path.join('data', file.split('.')[0])
                fc = FrameCounter(args)

                if args.load:
                    load_pickles(args, pickle_base, fc)
                else:
                    fc.compare_frames()
                    fc.save_timestamps()
                    if not args.no_write:
                        write_pickles(args, pickle_base, fc)

# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a video frame by frame to identify screen changes.')
    parser.add_argument('--threshold', default=30, type=int, help='mse threshold value')
    parser.add_argument('--videos', default='originals', type=str, help='Video directory to process')
    parser.add_argument('--load', default=False, action='store_true', help='load a data array from pickle')
    parser.add_argument('--no_write', default=False, action='store_true', help='do not write new data to pickles')
    parser.add_argument('--mse', default=False, action='store_true', help='perform mse comparison')
    parser.add_argument('--edge', default=False, action='store_true', help='perform edge detection')
    parser.add_argument('--ssim', default=False, action='store_true', help='perform ssim comparison')
    parser.add_argument('--lag', default=16, type=int, help='Rounds of frame processing to wait before saving a frame change (due to loading lag time)')
    parser.add_argument('--freq', default=2, type=int, help='Process every Nth frame')
    args = parser.parse_args()
    main(args)
