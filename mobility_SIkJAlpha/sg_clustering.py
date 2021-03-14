import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from matplotlib.dates import DateFormatter
import json
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import SI

data_file = "./data/sg/all_hourly_vis_till_week_2020-12-21.json"
distribution = "./data/sg/clustering/distribution.json"
clustering = "./data/sg/clustering/clustering.json"
loc_names = "./data/sg/location_name_lookup.json"
cluster_mobility = "./data/sg/clustering/cluster_mobility.pickle"
cluster_desc = "./data/sg/clustering/clustering_description.txt"
cluster_processed_dimension = "./data/sg/clustering/dimension.json"



total = pd.date_range(start = "2020-02-10T00:00:00", freq='1H', end="2020-12-27T23:00:00")
day_starts = pd.date_range(start = "2020-02-10T00:00:00", freq='1D', end="2020-12-27T00:00:00")
week_starts = pd.date_range(start = "2020-02-10", freq='7D', end="2020-12-21")
week = pd.date_range(start = "2020-02-17T00:00:00", freq="1H", end = "2020-02-23T23:00:00")



N = len(total)
# Number of clusters
CCNT = 3

def extract_one_week_distribution():
    sample_sz = 24*7
    data = json.load(open(data_file, 'r'))
    gc.collect()


    pbar = tqdm(desc= "Calculating a weekly distribution", total = len(data.keys()))

    for id, arr in data.items():
        data[id] = normalize(np.array(data[id][0:sample_sz]).reshape(1,sample_sz))[0]
        data[id] = list(data[id])
        pbar.update()

    with open(distribution, 'w') as f:
        json.dump(data,f)


def hierarchical_cluster():
    dis = json.load(open(distribution, 'r'))
    ids = []
    viss= []
    for id, arr in dis.items():
        ids.append(id)
        viss.append(arr)


    clst = AgglomerativeClustering(n_clusters=CCNT).fit_predict(viss)

    res = dict()
    for i in range(len(clst)):
        res[ids[i]] = int(clst[i])

    with open(clustering, 'w') as f:
        json.dump(res, f)




def gaussian_mixture_cluster():
    dis = json.load(open(distribution, 'r'))
    ids = []
    viss = []
    for id, arr in dis.items():
        ids.append(id)
        viss.append(arr)

    clst = GaussianMixture(n_components=CCNT, random_state=0).fit_predict(viss)

    res = dict()
    for i in range(len(clst)):
        res[ids[i]] = int(clst[i])

    with open(clustering, 'w') as f:
        json.dump(res, f)


def check_cluster_names():
    name = json.load(open(loc_names, 'r'))
    clst = json.load(open(clustering, 'r'))
    t = [[] for i in range(CCNT)]
    for id, cls in clst.items():
        t[cls].append(name[id])


    with open(cluster_desc, 'w') as f:
        for x in range(CCNT):
            f.write("Class " + str(x) + " has a total of " + str(len(t[x])) + " locations" + "\n")
            for nm in t[x]:
                f.write('\t'+ nm + '\n')
            f.write('\n\n\n\n\n')






def check_hourly_mobilty_distribution():
    tt = [np.zeros(N) for i in range(CCNT)]
    clst = json.load(open(clustering, 'r'))
    vis = json.load(open(data_file, 'r'))
    for id, arr in vis.items():
        cls = clst[id]
        tt[cls] = tt[cls] + np.array(arr)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(CCNT):
        one_cluster = pd.Series(tt[i], index=total)
        plt.plot(one_cluster[30:80], label="Class " + str(i))


    plt.title("Hourly Visitor Count Distribution", fontsize=30)
    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Cumulative Hourly Visitor Count", fontsize=28)
    sparse_tick = pd.date_range(start = total[30], end= total[80], freq="6H")
    print(sparse_tick)
    plt.xticks(sparse_tick)
    date_form = DateFormatter("%H:00")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(True)
    plt.legend(loc=0, fontsize=25)
    plt.show()


def check_daily_mobility_distribution():
    tt = [np.zeros(N) for i in range(CCNT)]
    clst = json.load(open(clustering, 'r'))
    vis = json.load(open(data_file, 'r'))
    for id, arr in vis.items():
        cls = clst[id]
        tt[cls] = tt[cls] + np.array(arr)

    for i in range(CCNT):
        hourly_vis = pd.Series(tt[i], index=total)
        vals = []
        for day in day_starts:
            hours = pd.date_range(start = day, freq='1H', periods=24)
            sm = 0
            for h in hours:
                sm += hourly_vis[h]
            vals.append(sm)
        one_cluster = pd.Series(vals, index=day_starts)
        one_cluster = SI.SI.kz_smooth(one_cluster,7)


        plt.plot(one_cluster, label="Class " + str(i))
    plt.legend(loc=0)
    plt.show()


def check_daily_mobility_distribution_of_tau():
    tt = [np.zeros(N) for i in range(CCNT)]
    vis = pickle.load(open(cluster_mobility, 'rb'))

    for i in range(CCNT):
        one_cluster = vis[i]
        one_cluster = SI.SI.kz_smooth(one_cluster,7)
        plt.plot(one_cluster, label="Class " + str(i))
    plt.legend(loc=0)
    plt.show()

def extract_mobility_from_clustering():
    # Load clustering

    def square(lst):
        arr = np.array(lst)
        rhs = arr - 1
        # To avoid -1
        rhs[rhs < 0] = 0
        return arr * rhs



    tt = [np.zeros(N) for i in range(CCNT)]
    clst = json.load(open(clustering, 'r'))
    vis = json.load(open(data_file, 'r'))
    for id, arr in vis.items():
        cls = clst[id]
        # n square
        tt[cls] = tt[cls] + square(arr)

    res = dict()

    for i in range(CCNT):
        # Convert to daily
        hourly_vis = pd.Series(tt[i], index=total)
        vals = []
        for day in day_starts:
            hours = pd.date_range(start=day, freq='1H', periods=24)
            sm = 0
            for h in hours:
                sm += hourly_vis[h]
            vals.append(sm)
        one_cluster = pd.Series(vals, index=day_starts)
        res[i] = one_cluster
    with open(cluster_mobility, 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

def how_many_POIs():
    vis = json.load(open(data_file, 'r'))
    print(len(vis.keys()))

check_hourly_mobilty_distribution()

# check_hourly_mobilty_distribution()
# how_many_POIs()
# check_daily_mobility_distribution()
# check_daily_mobility_distribution_of_tau()
# check_cluster_names()
# hierarchical_cluster()
# extract_one_week_distribution()
# extract_mobility_from_clustering()