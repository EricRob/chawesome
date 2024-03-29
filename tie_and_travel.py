import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import pdb
import os
import random
from colorsys import hsv_to_rgb
from math import isnan


LEGEND = []
COLORS = {}
SCREEN_TO_COLOR = {}


# FUNCTIONS
def sort_screens(*csvs):
    macros = pd.read_csv('csvs/screen_info.csv')
    for index, csv_name in enumerate(csvs):
        times = pd.read_csv(csv_name)
        for i, row in times.iterrows():
            if isnan(row.Screen):
                continue
            else:
                try:
                    category = macros.loc[macros['ID'] == str(int(row.Screen)), 'MACRO'].values
                except:
                    pdb.set_trace()
                if len(category) > 0:
                    category = category[0]
                else:
                    category = 'UNKNOWN CATEGORY'
                if category not in COLORS:
                    COLORS[category] = []
                if row.Screen not in COLORS[category]:
                    COLORS[category].append(int(row.Screen))
    
    hue_increment = 360 // (len(COLORS) + 1)
    hue = 0
    for index, category in enumerate(COLORS):
        COLORS[category].sort()
        hue += hue_increment*2
        print(hue)
        saturation = np.linspace(0.6, 1, num=len(COLORS[category]))
        value = np.linspace(.6, .90,  num=len(COLORS[category]))
        print(category)
        for i, screen in enumerate(COLORS[category]):
            h = hue/360
            s = saturation[i]
            v = value[i]

            # Mult by 255?
            r, g, b = hsv_to_rgb(h, s, v)
            # r, g, b = [int(x) for x in (r, g, b)]
            r, g, b = [int(x*255) for x in (r, g, b)]
            SCREEN_TO_COLOR[str(screen)] = '#%02X%02X%02X' % (r, g, b)

def tie_plot(*csvs, name):

    for index, csv_name in enumerate(csvs):
        times = pd.read_csv(csv_name)
        times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
        times = times.sort_values(by='end', ascending=False)

        height = 0.3

        for i, row in times.iterrows():
            if isnan(row.Screen):
                continue
            screen = str(int(row.Screen))
            plt.barh(index + 1, row.end, color=SCREEN_TO_COLOR[screen], height=height)

    plt.plot()
    plt.xlabel("Elapsed time")
    plt.ylabel("Subject")
    plt.title(f"{name.capitalize()} Screen switches")
    plt.savefig(os.path.join('data', f'{name}_tie_plot.png'))
    plt.clf()

def travel_graph(*csvs, name='all', w=12, h=20, ns=50):
    fig, ax = plt.subplots(nrows=9, ncols=4, figsize=(w,h))
    # fig, ax = plt.subplots(3, 3, num=1)
    G = nx.DiGraph()
    edges = {}

    for category in COLORS:
        for screen in COLORS[category]:
            G.add_node(screen)
    for index, csv_name in enumerate(csvs):
        label = csv_name.split('.')[0].split('/')[-1]
        edges[label] = []
        times = pd.read_csv(csv_name)
        times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
        times = times.sort_values(by='end', ascending=False)

        a = None
        for i, row in times.iterrows():
            if isnan(row.Screen):
                continue
            else:
                b = int(row.Screen)
                if a:
                    edges[label].append((a, b))
                a = b

    color_arr = []
    for node in G.nodes():
        color_arr.append(SCREEN_TO_COLOR[str(node)])

    pos = nx.circular_layout(G)

    r = lambda: random.randint(0,255)
    
    for index, edge in enumerate(edges):
        axis = ax[1 + index % 8][1 + index // 8]
        # axis.set_title(edge)
        axis.axis('off')
        plt.sca(axis)
        color = '#%02X%02X%02X' % (r(),r(),r())
        color = '#181818'
        grey_color = '#808080'
        nx.draw_networkx_nodes(G, pos, ax=axis, node_color=color_arr, node_size=ns)
        # nx.draw_networkx_nodes(G, pos, ax=axis, node_color=grey_color, node_size=ns)
        nx.draw_networkx_edges(G, pos, ax=axis, edgelist=edges[edge], edge_color=color, arrows=True)
        # nx.draw_networkx_labels(G, pos, ax=axis, )
        # plt.title(edge)
    graph_legend = []
    for category in COLORS:
        graph_legend.append(plt.Line2D([0], [0], color=SCREEN_TO_COLOR[str(COLORS[category][-1])], lw=3, label=category))
    axis = ax[0][0]
    axis.axis('off')
    axis.legend(handles=graph_legend, loc='center', fancybox=True, shadow=True)
    
    for x in range(8):
        axis = ax[x + 1][0]
        axis.axis('off')
        axis.text(0.5, 0.5, f'Pharmacist {x + 1}')
    for x in range(3):
        axis = ax[0][x + 1]
        axis.axis('off')
        axis.text(0.5, 0.5, f'Case {x+1}')

    # plt.savefig(os.path.join('data', f'{name}_circ_graphs.png'))
    plt.savefig(os.path.join('data', f'w{w}_h{h}_ns{ns}_all_cases_circ_graphs.png'))
    plt.clf()


# file handler
names = ['helen', 'laura', 'malia']
all_csvs = []

for name in names:
    for i in range(8):
        all_csvs.append(os.path.join('csvs', f'{name}_{i + 1}.csv'))
sort_screens(*all_csvs)

for name in names:
    csvs = []
    for i in range(8):
        csvs.append(os.path.join('csvs', f'{name}_{i + 1}.csv'))
    tie_plot(*csvs, name=name)

# for name in names:
#     csvs = []
#     for i in range(8):
#         csvs.append(os.path.join('csvs', f'{name}_{i + 1}.csv'))
#     travel_graph(*csvs, name=name)

# travel_graph(*all_csvs)
# travel_graph(*all_csvs, w=15, h=30)
travel_graph(*all_csvs, w=15, h=30, ns=40)
travel_graph(*all_csvs, w=15, h=30, ns=30)
travel_graph(*all_csvs, w=15, h=30, ns=20)

