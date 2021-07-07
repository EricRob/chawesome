import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import pdb
import random


LEGEND = []
COLORS = {}

# FUNCTIONS
def tie_plot(csv_name, index):
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

    for i, row in times.iterrows():
        screen = str(int(row.Screen))
        if screen not in LEGEND:
            plt.barh(index, row.end, color=COLORS[screen], label=screen, height=height)
            LEGEND.append(screen)
        else:
            plt.barh(index, row.end, color=COLORS[screen], height=height)


def travel_graph(csv_name, index, ax, gtype):

    times = pd.read_csv(csv_name)
    times.loc[:, 'end'] = times['end'].apply(lambda x: x - times['start'].min())
    times = times.sort_values(by='end', ascending=False)

    edges = []
    a = None
    color_arr = []
    for i, row in times.iterrows():
        b = int(row.Screen)
        if a:
            edges.append((a, b))
        a = b

    G = nx.DiGraph()
    G.add_edges_from(edges)

    for node in G.nodes():
        color_arr.append(COLORS[str(node)])

    if gtype == 'spring':
        pos = nx.spring_layout(G)
    elif gtype == 'circular':
        pos = nx.circular_layout(G)
    elif gtype == 'planar':
        try:
            pos = nx.planar_layout(G)
        except:
            return

    axis = ax[index // 3][index % 3]
    plt.sca(axis)
    nx.draw_networkx_nodes(G, pos, node_color=color_arr, ax=axis, node_size=100)
    nx.draw_networkx_labels(G, pos, ax=axis)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, ax=axis)


# MAIN

csvs = ['heath_helen.csv', 'jason_helen.csv']

for index, csv in enumerate(csvs):
    tie_plot(csv, index)

plt.plot()
plt.xlabel("Elapsed time")
plt.ylabel("Subject")
plt.title("Helen Screen switches")
#plt.legend(loc='upper right')
plt.legend(loc='right', bbox_to_anchor=(1.2, 0.5), fancybox=True, shadow=True, ncol=1)
plt.tight_layout()
plt.show()
plt.clf()

fig, ax = plt.subplots(2, 3, num=1)

for index, csv in enumerate(csvs):
    travel_graph(csv, index*3, ax, 'spring')
    travel_graph(csv, index*3 + 1, ax, 'circular')
    travel_graph(csv, index + 2, ax, 'planar')
plt.show()
