import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import pdb
import os
import random


LEGEND = []
COLORS = {}

# FUNCTIONS
def tie_plot(*csvs):
    for index, csv_name in enumerate(csvs):
        times = pd.read_csv(csv_name)
        times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
        times = times.sort_values(by='end', ascending=False)

        height = 0.3

        screens = times.Screen.unique()
        r = lambda: random.randint(0,255)
        for screen in screens:
            scr = str(screen)
            if scr not in COLORS:
                COLORS[scr] = '#%02X%02X%02X' % (r(),r(),r())
        if index == 4:
            continue
        for i, row in times.iterrows():
            screen = str(int(row.Screen))
            if screen not in LEGEND:
                plt.barh(index, row.end, color=COLORS[screen], label=screen, height=height)
                LEGEND.append(screen)
            else:
                plt.barh(index, row.end, color=COLORS[screen], height=height)

    plt.plot()
    plt.xlabel("Elapsed time")
    plt.ylabel("Subject")
    plt.title("Helen Screen switches")
    plt.savefig(os.path.join('data', 'tie_plot.png'))
    plt.legend(loc='center', fancybox=True, shadow=True, ncol=3)
    plt.savefig(os.path.join('data', 'legend.png'))

def travel_graph(*csvs):
    fig, ax = plt.subplots(3, 3, num=1)
    fig.set_size_inches(12, 12)
    G = nx.DiGraph()
    edges = {}

    for index, csv_name in enumerate(csvs):
        label = csv_name.split('.')[0].split('/')[-1]
        edges[label] = []
        times = pd.read_csv(csv_name)
        times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
        times = times.sort_values(by='end', ascending=False)

        a = None
        for i, row in times.iterrows():
            b = int(row.Screen)
            G.add_node(b)
            if a:
                edges[label].append((a, b))
            a = b

        
        # G.add_edges_from(edges)
    color_arr = []
    for node in G.nodes():
        color_arr.append(COLORS[str(node)])

    pos = nx.circular_layout(G)
    # if gtype == 'spring':
    #     pos = nx.spring_layout(G)
    # elif gtype == 'circular':
    #     pos = nx.circular_layout(G)
    # elif gtype == 'planar':
    #     try:
    #         pos = nx.planar_layout(G)
    #     except:
    #         return

    # 
    # plt.sca(axis)
    r = lambda: random.randint(0,255)
    
    for index, edge in enumerate(edges):
        axis = ax[index // 3][index % 3]
        axis.set_title(edge)
        plt.sca(axis)
        color = '#%02X%02X%02X' % (r(),r(),r())
        nx.draw_networkx_edges(G, pos, ax=axis, edgelist=edges[edge], edge_color=color, arrows=True)
        nx.draw_networkx_nodes(G, pos, ax=axis, node_color=color_arr, node_size=100)
        nx.draw_networkx_labels(G, pos, ax=axis, )
        plt.title(edge)
    plt.savefig(os.path.join('data', 'circ_graphs.png'))


# MAIN

# csvs = ['heath_helen.csv', 'jason_helen.csv']

# for index, csv in enumerate(csvs):
#     tie_plot(csv, index)


# file handler
csvs = []
for i in range(8):
    csvs.append(os.path.join('data', 'csvs', f'helen_{i + 1}.csv'))


tie_plot(*csvs)
travel_graph(*csvs)

# for index, csv in enumerate(csvs):
#     travel_graph(csv, index*3, ax, 'spring')
#     travel_graph(csv, index*3 + 1, ax, 'circular')
#     travel_graph(csv, index + 2, ax, 'planar')
# plt.show()
