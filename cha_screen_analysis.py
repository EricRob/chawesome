#!/user/bin/env python3 -tt
"""
Organize timestamp and saved frame data from screen
recording of GE's Centricity High Actuity (CHA) to
create usage summaries and a travel graph.

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
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from IPython.display import display
from math import sqrt, floor, ceil

# Global variables

# Class declarations

class NetworkGraph(object):
    """docstring for NetworkGraph"""
    def __init__(self, cha_summary):
        self.cs = cha_summary
        self.nodes = {}
        self.node_path = []
        self.node_sizes = {}
        self.edges = []
        self.build_nodes()
        self.build_edges()

    def visualize(self, layout):
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        sizes = []
        for node in G.nodes():
            sizes.append(self.node_sizes[node]*30)
        if layout == 'spring':
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = sizes)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edgelist=self.edges, arrows=True)
            plt.savefig(self.cs.graph_path, format="PNG")
        elif layout == 'circular':
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = sizes)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edgelist=self.edges, arrows=True)
            plt.savefig(os.path.join(self.cs.base_path, 'circular_graph.png'), format="PNG")
        elif layout == 'planar':
            pos = nx.planar_layout(G)
            nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = sizes)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edgelist=self.edges, arrows=True)
            plt.savefig(os.path.join(self.cs.base_path, 'planar_graph.png'), format="PNG")

        plt.clf()
        return

    def build_nodes(self):
        for group_id in self.cs.frame_groups:
            group = self.cs.frame_groups[group_id]
            node_id = str(self.cs.time_dict[str(group[0])])
            node = NetworkNode(group, node_id)
            self.node_sizes[node_id] = node.size
            self.nodes[node_id] = node

    def build_edges(self):
        start = None
        for time in self.sc.timestamps:
            node_id = self.find_next_node(time)
            self.node_path.append(node_id)
            if start:
               self.edges.append((start, node_id))
            start = node_id

    def find_next_node(self,time):
        for node in self.nodes:
            if time in self.nodes[node].times:
                return node


class NetworkNode(object):
    """docstring for Node"""
    def __init__(self, group, node_id):
        self.id = node_id
        self.size = len(group)
        self.times = sorted(group)
        self.reference = ''     
        
        

class ChaSummary(object):
    """docstring for ChaSummary"""
    def __init__(self, args):
        self.args = args
        self.video_name = args.video
        self.base_name = self.video_name.split('/')[-1].split('.')[0]
        self.threshold = args.threshold
        self.base_path = os.path.join('data', self.base_name)
        self.output_path = os.path.join('outputs', 'output_{}.mp4'.format(self.base_name))
        self.frames_dir = os.path.join(self.base_path, 'areas_of_interest')
        self.graph_path = os.path.join(self.base_path,f'{self.base_name}_graph.png')
        os.makedirs(self.frames_dir, exist_ok=True)

        self.timestamp_path = os.path.join(self.base_path, 'timestamps.txt')
        self.timestamps = []
        self.time_dict = {}
        self.frame_groups = {}
        self.screen_duration = []
        self.screen_durations = {}
        self.avg_screen_durations = []

        self.scrolling_frames = []
        self.scrolling_times = []

        ## Constants
        # seconds
        self.scroll_threshold = 1

        vid = cv2.VideoCapture(self.video_name)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

        default_width = 1080
        default_height = 720

        self.box_x = 0 * (width / default_width)
        self.box_y = 98 * (height / default_height)
        self.box_w = 743 * (width / default_width)
        self.box_h = 555 * (height / default_height)

        # no unit
        self.mse_threshold = 100
        

    def read_timestamps(self):
        frame_idx = 0
        with open(self.timestamp_path) as f:
            reader = csv.reader(f)
            for line in reader:
                frame_path = os.path.join(self.frames_dir,'{:06d}.jpg'.format(int(line[1])))
                if os.path.exists(frame_path):
                    frame_idx = int(line[1])
                    time = float(line[0])
                    self.timestamps.append(time)
                    self.time_dict[str(time)] = frame_idx

        self.total_frames = frame_idx

    def set_timestamp_path(self, path):
        self.timestamp_path = path

    def mse(self, imageA, imageB):
        # Mean Standard Error
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def pull_frame(self, timestamp):
        frame_path = os.path.join(self.frames_dir,'{:06d}.jpg'.format(self.time_dict[str(timestamp)]))
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def remove_dissimilar_frames(self, errs):
        similars = []
        pdb.set_trace()
        for frame in errs:
            if frame[1] < self.mse_threshold:
                similars.append(frame[0])
        return similars

    def frame_previously_compared(self, groups, timestamp):
        for group in groups:
            if isinstance(groups, dict):
                for time in groups[group]:
                    if time is timestamp:
                        return True
            elif isinstance(groups, list):
                for time in groups:
                    if time is timestamp:
                        return True
        return False

    def compare_frames(self):
        for i in self.timestamps:
            if not self.frame_previously_compared(self.frame_groups, i):
            # if not self.frame_previously_compared(self.frame_groups, self.time_dict[str(i)]):
                errs = []
                frame1 = self.pull_frame(i)
                for j in self.timestamps:
                    if i < j:
                        frame2 = self.pull_frame(j)
                        err = self.mse(frame1, frame2)
                        errs.append((j, err))
                        # errs.append((self.time_dict[str(j)], err))
                errs.sort(key=lambda tup: tup[1])
                matches = self.remove_dissimilar_frames(errs)
                matches.append(i)
                # matches.append(self.time_dict[str(i)])
                self.frame_groups[self.time_dict[str(matches[0])]] = matches

    def get_frame_group_from_timestamp(self, time):
        for group in self.frame_groups:
            if time in self.frame_groups[group]:
                return group


    def time_per_frame(self):
        start_times = np.array(self.timestamps)
        end_times = np.roll(np.array(self.timestamps), -1)
        self.screen_duration = np.around(end_times - start_times, decimals=2)
        screens_zipped = list(zip(self.timestamps, self.screen_duration))
        for pair in screens_zipped:
            if pair[1] < 0:
                continue
            frame_group = self.get_frame_group_from_timestamp(pair[0])
            if frame_group not in self.screen_durations:
                self.screen_durations[frame_group] = []
            self.screen_durations[frame_group].append(pair[1])
        


    def create_table(self):
        nrows = len(self.screen_durations)
        columns = ['frame', 'count', 'mean', 'median', 'std']
        df = pd.DataFrame(columns=columns)
        for group in self.screen_durations:
            times = self.screen_durations[group]
            df = df.append({
                'frame' : str(group),
                'count' : len(times),
                'total time': round(np.sum(times),1),
                'mean'  : np.mean(times),
                'median': np.median(times),
                'std'   : np.std(times)
                }, ignore_index = True)
        df.sort_values(by=['count'], ascending=False, inplace=True)
        self.csv_path = os.path.join(self.base_path,f"{self.base_name}_summary.csv")
        df.to_csv(self.csv_path)

    def plot_dims(self, size):
        s = sqrt(size)
        l = floor(s)
        w = ceil(size / l)
        return w, l

    def frame_grid(self):
        self.shrink_timestamps()
        size = len(self.timestamps)
        width, length = self.plot_dims(size)
        img_x = int(216*1.5)
        img_y = int(144*1.5)
        depth = 3
        dsize = (img_x, img_y)
        
        final_board = np.zeros((length*img_y, width*img_x, depth))
        for i in np.arange(length):
            row = np.zeros((img_y, width*img_x, depth))
            for j in np.arange(width):
                loc = (i-1)*length + j + (size // length) + i
                if loc == size:
                    break
                img = self.grid_frame(self.timestamps[loc])
                img = cv2.resize(img, dsize=dsize)
                xlim1 = j * img_x
                xlim2 = xlim1 + img_x
                row[:,xlim1:xlim2,:] = img
            ylim1 = i * img_y
            ylim2 = ylim1 + img_y
            final_board[ylim1:ylim2,:,:] = row
        cv2.imwrite(os.path.join(self.base_path, 'frame_path.jpg'), final_board)

    def grid_frame(self, timestamp):
        # pdb.set_trace()
        frame_path = os.path.join(self.base_path, 'single_frames','{:06d}.jpg'.format(self.time_dict[str(timestamp)]))
        frame = cv2.imread(frame_path)
        return frame

    def shrink_timestamps(self):
        #I'm sorry, this is a hack.
        remove = []
        for time in self.timestamps:
            frame_path = os.path.join(self.base_path, 'single_frames','{:06d}.jpg'.format(self.time_dict[str(time)]))
            if not os.path.exists(frame_path):
                remove.append(time)
        for time in remove:
            self.timestamps.remove(time)
        
# Function declarations

def main(args):
    # config = CounterConfig(args)
    for _, _, files in os.walk(args.videos):
        for file in files:
            if file.endswith('.mp4'):
                pickle_path = os.path.join('data', file.split('.')[0], 'sc_pickle.obj')
                if os.path.exists(pickle_path) and args.load:
                    with open(pickle_path, 'rb') as f:
                        print(f'loading pickle {pickle_path}...')
                        cs = pickle.load(f)
                else:
                    args.video = os.path.join(args.videos,file)
                    cs = ChaSummary(args)
                    cs.read_timestamps()
                    cs.compare_frames()
                    with open(pickle_path, 'wb') as f:
                        print(f'writing pickle {pickle_path}...')
                        pickle.dump(cs, f)
                cs.time_per_frame()
                cs.create_table()
                cs.frame_grid()

                g = NetworkGraph(cs)
                g.visualize('spring')
                g.visualize('circular')
                g.visualize('planar')
    

# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify total and unique screens from an application video, potentially create a travel graph.')
    parser.add_argument('--threshold', default=30, type=int, help='Pixel percentage change for screen changes')
    parser.add_argument('--videos', default='originals', type=str, help='Name of directory to scan for summary')
    parser.add_argument('--load', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
