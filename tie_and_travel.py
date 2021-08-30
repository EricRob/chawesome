import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import pdb
import os
import random
from colorsys import hsv_to_rgb


LEGEND = []
COLORS = {}
SCREEN_TO_COLOR = {}


# FUNCTIONS
def sort_screens(*csvs):
    macros = pd.read_csv('csvs/screen_info.csv')
    for index, csv_name in enumerate(csvs):
        times = pd.read_csv(csv_name)
        for i, row in times.iterrows():
            category = macros.loc[macros['ID'] == str(row.Screen), 'MACRO'].values
            if len(category) > 0:
                category = category[0]
            else:
                category = 'UNKNOWN CATEGORY'
            if category not in COLORS:
                COLORS[category] = []
            if row.Screen not in COLORS[category]:
                COLORS[category].append(row.Screen)
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

def tie_plot(*csvs):

    for index, csv_name in enumerate(csvs):
        times = pd.read_csv(csv_name)
        times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
        times = times.sort_values(by='end', ascending=False)

        height = 0.3

        for i, row in times.iterrows():
            screen = str(int(row.Screen))
            plt.barh(index, row.end, color=SCREEN_TO_COLOR[screen], height=height)

    # for category in COLORS:
    #     for screen in COLORS[category]:
    #         LEGEND.append(plt.Line2D([0], [0], color=SCREEN_TO_COLOR[str(screen)], lw=3, label=str(screen)))

    plt.plot()
    plt.xlabel("Elapsed time")
    plt.ylabel("Subject")
    plt.title("Helen Screen switches")
    plt.savefig(os.path.join('data', 'tie_plot.png'))
    # plt.legend(handles=LEGEND, loc='center', fancybox=True, shadow=True, ncol=3)
    # plt.savefig(os.path.join('data', 'legend.png'))
    plt.clf()

def travel_graph(*csvs):
    fig, ax = plt.subplots(3, 3, num=1)
    fig.set_size_inches(12, 12)
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
        axis = ax[index // 3][index % 3]
        axis.set_title(edge)
        plt.sca(axis)
        color = '#%02X%02X%02X' % (r(),r(),r())
        color = '#181818'
        nx.draw_networkx_edges(G, pos, ax=axis, edgelist=edges[edge], edge_color=color, arrows=True)
        nx.draw_networkx_nodes(G, pos, ax=axis, node_color=color_arr, node_size=50)
        # nx.draw_networkx_labels(G, pos, ax=axis, )
        plt.title(edge)
    graph_legend = []
    for category in COLORS:
        graph_legend.append(plt.Line2D([0], [0], color=SCREEN_TO_COLOR[str(COLORS[category][-1])], lw=3, label=category))
    axis = ax[2][2]
    axis.axis('off')
    axis.legend(handles=graph_legend, loc='center', fancybox=True, shadow=True)
    plt.savefig(os.path.join('data', 'circ_graphs.png'))


# MAIN

# csvs = ['heath_helen.csv', 'jason_helen.csv']

# for index, csv in enumerate(csvs):
#     tie_plot(csv, index)


# file handler
csvs = []
for i in range(8):
    csvs.append(os.path.join('csvs', f'helen_{i + 1}.csv'))

sort_screens(*csvs)
tie_plot(*csvs)
travel_graph(*csvs)

# for index, csv in enumerate(csvs):
#     travel_graph(csv, index*3, ax, 'spring')
#     travel_graph(csv, index*3 + 1, ax, 'circular')
#     travel_graph(csv, index + 2, ax, 'planar')
# plt.show()
