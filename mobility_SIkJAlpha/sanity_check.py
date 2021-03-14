import pandas as pd
import numpy as np
from safegraph_data_preprocess import to_py_list
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

SAMPLE_FILE = "./data/sg/2020-02-10/2020-02-10-weekly-patterns.csv"
CLUSTERING= "./data/sg/manual_clustering/clustering.json"
clustering = json.load(open(CLUSTERING, 'r'))
CLUSTERS = ["dining", "school", "shopping", "hotel", "recreation", "airport"]
common = json.load(open("./data/sg/common_ids_till_week_2020-12-21.json",'r'))

# 2020-02-10T00:00:00-08:00,2020-02-17T00:00:00-08:00
total = pd.date_range(start = "2020-02-10T00:00:00", freq='1H', end="2020-02-16T23:00:00")
for d in total:
    print(d)

hour_cnt = len(total)

df = pd.read_csv(SAMPLE_FILE, index_col='safegraph_place_id')

vis_cnt = dict()

pbar = tqdm(total=len(common), desc="Counting: ")
print(hour_cnt)
print(len(to_py_list(df.iloc[0]['visits_by_each_hour'])))

assert(len(to_py_list(df.iloc[0]['visits_by_each_hour']))==hour_cnt)


for x in common:
    print(x)
    vis_cnt[x] = to_py_list(df.loc[x]['visits_by_each_hour'])
    pbar.update()


res = dict()

for c in CLUSTERS:
    tt = np.zeros(168)
    for id in clustering[c]:
        tt += np.array(vis_cnt[id])
    res[c] = pd.Series(tt, index=total)

for c in CLUSTERS:
    plt.plot(res[c])

plt.title("Sanity Check")
plt.grid("True")
plt.show()



print(vis_cnt)


