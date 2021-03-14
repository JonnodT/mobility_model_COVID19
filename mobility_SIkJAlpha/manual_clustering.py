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

# Data files
vis_count_file = "./data/sg/all_hourly_vis_till_week_2020-12-21.json"
loc_names = "./data/sg/location_name_lookup.json"
common_id_file = "./data/sg/common_ids_till_week_2020-12-21.json"
address_file = "./data/sg/address_lookup.json"


#Output Files
cluster_desc = "./data/sg/manual_clustering/clustering_in_loc_names.txt"
clustering = "./data/sg/manual_clustering/clustering.json"
cluster_processed_dimension = "./data/sg/manual_clustering/dimension.json"


# Filters
dinning_filter_file = "./data/sg/manual_clustering/filters/dining.txt"
school_filter_file = "./data/sg/manual_clustering/filters/school.txt"
shopping_filter_file = "./data/sg/manual_clustering/filters/shopping.txt"
hotel_filter_file = "./data/sg/manual_clustering/filters/hotels.txt"
recreation_filter_file = "./data/sg/manual_clustering/filters/recreation.txt"
airport_filter_file = "./data/sg/manual_clustering/filters/airport.txt"

#Pre-defined date ranges
total = pd.date_range(start = "2020-02-10T00:00:00", freq='1H', end="2020-12-27T23:00:00")
day_starts = pd.date_range(start = "2020-02-10T00:00:00", freq='1D', end="2020-12-27T00:00:00")
week_starts = pd.date_range(start = "2020-02-10", freq='7D', end="2020-12-21")
lens = pd.date_range(start = "2020-02-11T06:00:00", freq="1H", end = "2020-02-13T08:00:00")


CLUSTERS = ["dining", "school", "shopping", "hotel", "recreation", "airport"]

N = len(total)
# Number of clusters
ccnt = len(CLUSTERS)


def read_set_of_keywords_from_file(fname):
    res = set()
    words =  open(fname,'r').readlines()
    for word in words:
        res.add(word.strip())
    return res



def generate_cluster_based_on_filters():

    # load filters
    dining = read_set_of_keywords_from_file(dinning_filter_file)
    school = read_set_of_keywords_from_file(school_filter_file)
    shopping = read_set_of_keywords_from_file(shopping_filter_file)
    hotel = read_set_of_keywords_from_file(hotel_filter_file)
    recreation = read_set_of_keywords_from_file(recreation_filter_file)
    airport = read_set_of_keywords_from_file(airport_filter_file)
    filter_dict = {"dining" : dining, "school":school, "shopping":shopping, "hotel":hotel, "recreation":recreation, "airport":airport}
    vis_count = json.load(open(vis_count_file, 'r'))



    for x in school:
        print(x)

    # Load all ids to be cluster and name and address
    sgids = json.load(open(common_id_file, 'r'))
    name_dict = json.load(open(loc_names, 'r'))
    addr_dict = json.load(open(address_file, 'r'))



    res_dict = {"dining" : [], "school":[], "shopping":[], "hotel":[], "recreation":[], "airport":[]}
    cnt = 0
    tt = len(sgids)


    for id in sgids:
        cnt+=1
        curr_name = name_dict[id].split(" ")
        belongs_to = []
        for filtername, filterwords in filter_dict.items():
            ok = False
            for w in filterwords:
                if w in curr_name:
                    ok = True
                    break
            if ok:
                belongs_to.append(filtername)
        if len(belongs_to) != 1:
            curr_name = name_dict[id]
            total_vis = np.sum(vis_count[id])
            if total_vis > 7000:
                curr_addr = addr_dict[id]
                print("Please manually classify location: " + curr_name)
                print("Located at: " + curr_addr)
                print("It has a total visitor count of: " + str(total_vis))
                print("By the automatic filter, it belongs to:")
                print("Clustered " + str(cnt) + " out of " + str(tt))
                if(len(belongs_to) == 0):
                    print("\tNone")
                else:
                    for clst in belongs_to:
                        print("\t" + clst)
                entered_cluster = ""
                while entered_cluster not in res_dict.keys() and entered_cluster != "skip":
                    if entered_cluster == "skip":
                        break
                    print("Enter the cluster it belongs to: ")
                    entered_cluster = input()
                if entered_cluster != "skip":
                    res_dict[entered_cluster].append(id)


        else:
            res_dict[belongs_to[0]].append(id)


    with open(clustering, 'w') as f:
        json.dump(res_dict, f)


    with open(cluster_desc,'w') as f:
        for name, arr in res_dict.items():
            f.write("\n\nCluster: " + name + "\n")
            for id in arr:
                f.write(name_dict[id] + "\n")


def check_clustering_coverage():
    vis_count = json.load(open(vis_count_file, 'r'))
    clusters = json.load(open(clustering,'r'))

    clustered_ids = set()
    for c in CLUSTERS:
        for id in clusters[c]:
            clustered_ids.add(id)

    tt = 0
    clsted = 0
    for id in vis_count:
        sm = np.sum(vis_count[id])
        if id in clustered_ids:
            clsted += sm
        tt += sm
    print(clsted / tt)


def check_clustering_daily_distribution():
    vals = [np.zeros(N) for i in range(len(CLUSTERS))]
    hourly_sm = dict(zip(CLUSTERS, vals))
    daily_sm = dict(zip(CLUSTERS, vals))



    vis = json.load(open(vis_count_file, 'r'))
    clst = json.load(open(clustering, 'r'))

    # Aggregation
    for c in CLUSTERS:
        for id in clst[c]:
            raw = np.array(vis[id])
            raw *= raw
            hourly_sm[c] += raw

    #Change to hourly time series
    for c in CLUSTERS:
        hourly_sm[c] = pd.Series(hourly_sm[c], index=total)

    for c in CLUSTERS:
        vals = []
        for day in day_starts:
            hours = pd.date_range(start=day, freq='1H', periods=24)
            sm = 0
            for h in hours:
                sm += hourly_sm[c][h]
            vals.append(sm)
        curr_daily_sm = pd.Series(vals, index=day_starts)
        curr_daily_sm = SI.SI.kz_smooth(curr_daily_sm,15)
        daily_sm[c] = curr_daily_sm

        plt.plot(curr_daily_sm, label=c)
    plt.title("Mobility of manual clusters (smooth window size = 15)")
    plt.legend(loc=0)
    plt.show()


def check_clustering_houly_distribution():
    # Change global variable $lens$ to tweak the window of inspection
    fig, ax = plt.subplots(figsize=(12, 8))
    vals = [np.zeros(N) for i in range(len(CLUSTERS))]
    hourly_sm = dict(zip(CLUSTERS, vals))
    daily_sm = dict(zip(CLUSTERS, vals))

    vis = json.load(open(vis_count_file, 'r'))
    clst = json.load(open(clustering, 'r'))

    # Aggregation
    for c in CLUSTERS:
        for id in clst[c]:
            hourly_sm[c] += vis[id]

    #Change to hourly time series
    for c in CLUSTERS:
        hourly_sm[c] = pd.Series(hourly_sm[c], index=total)

    for c in CLUSTERS:
        vals = []
        for h in lens:
            vals.append(hourly_sm[c][h])
        lens_series = pd.Series(vals, index=lens)

        plt.plot(lens_series, label=c)

    plt.title("Hourly Visitor Count Distribution", fontsize=30)
    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Cumulative Hourly Visitor Count", fontsize=28)
    sparse_tick = pd.date_range(start=lens[0], end=lens[-1], freq="6H")
    plt.xticks(sparse_tick)
    date_form = DateFormatter("%H:00")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(True)
    plt.legend(loc=0, fontsize=20)
    plt.show()

'''
plt.title("Hourly Visitor Count Distribution", fontsize=30)
    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Cumulative Hourly Visitor Count", fontsize=28)
    sparse_tick = pd.date_range(start = total[30], end= total[80], freq="4H")
    print(sparse_tick)
    plt.xticks(sparse_tick)
    date_form = DateFormatter("%H:00")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(loc=0, fontsize=25)
    plt.show()
'''

def generate_manual_cluster_daily_mobility_dimension():
    def square(a):
        # n * (n-1) / 2
        b = a - 1
        return (a * b) / 2

    vals = [np.zeros(N) for i in range(len(CLUSTERS))]
    hourly_sm = dict(zip(CLUSTERS, vals))
    daily_sm = dict()

    vis = json.load(open(vis_count_file, 'r'))
    clst = json.load(open(clustering, 'r'))

    # Aggregation
    for c in CLUSTERS:
        for id in clst[c]:
            raw = np.array(vis[id])
            raw = square(raw)
            hourly_sm[c] += raw

    # Change to hourly time series
    for c in CLUSTERS:
        hourly_sm[c] = pd.Series(hourly_sm[c], index=total)

    for c in CLUSTERS:
        vals = []
        for day in day_starts:
            hours = pd.date_range(start=day, freq='1H', periods=24)
            sm = 0
            for h in hours:
                sm += hourly_sm[c][h]
            vals.append(sm)
        daily_sm[c] = vals

    with open(cluster_processed_dimension, 'w') as f:
        json.dump(daily_sm, f)




# check_clustering_coverage()
# generate_cluster_based_on_filters()
# check_clustering_daily_distribution()
check_clustering_houly_distribution()

# generate_manual_cluster_daily_mobility_dimension()


