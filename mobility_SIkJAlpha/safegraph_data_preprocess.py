import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import ast
import json
import pickle
import operator

cities = ["Agoura Hills","Alhambra","Arcadia","Artesia","Avalon","Azusa","Baldwin Park","Bell","Bell Gardens","Bellflower","Beverly Hills","Bradbury","Burbank","Calabasas","Carson","Cerritos","Claremont","Commerce","Compton","Covina","Cudahy","Culver City","Diamond Bar","Downey","Duarte","El Monte","El Segundo","Gardena","Glendale","Glendora","Hawaiian Gardens","Hawthorne","Hermosa Beach","Hidden Hills","Huntington Park","Industry","Inglewood","Irwindale","La Cañada Flintridge","La Habra Heights","La Mirada","La Puente","La Verne","Lakewood","Lancaster","Lawndale","Lomita","Long Beach","Los Angeles","Lynwood","Malibu","Manhattan Beach","Maywood","Monrovia","Montebello","Monterey Park","Norwalk","Palmdale","Palos Verdes Estates","Paramount","Pasadena","Pico Rivera","Pomona","Rancho Palos Verdes","Redondo Beach","Rolling Hills","Rolling Hills Estates","Rosemead","San Dimas","San Fernando","San Gabriel","San Marino","Santa Clarita","Santa Fe Springs","Santa Monica","Sierra Madre","Signal Hill","South El Monte","South Gate","South Pasadena","Temple City","Torrance","Vernon","Walnut","West Covina","West Hollywood","Westlake Village","Whittier"]
exclude = {'Walnut Grove', 'Walnut Creek', 'Bella Vista', 'Terra Bella', 'Bell Canyon'}
loc_names = "./data/sg/location_name_lookup.json"
filter = {'Target','Airport','Shopping',"McDonald's",'Walmart','Depot','Costco','Wholesale','Hospital','Chevron','CVS','Shell','Chick-fil-A',"Carl's",'UCLA',"Joe's","Wendy's",'Park', 'Restaurant', 'School','Church', 'Bar', 'Cafe','Bar','LAX','Auto','Beach','Park','Tire','Repair','Health','Medical','Pet','Christian','Elementary','BBQ','Thai','Deli','Tea','Pho','Grill','Pet','Baptist','Flowers','Liquor','Smoke','Furniture','Hotel','Inn','Motel','Pet','Medical','University','Store','Preschool','Yoga','Pilates','Grill', 'Pizza', 'Bakery','Donut','Seafood','Library','Museum', 'Burgers', 'Inn', 'Pharmacy','Hotel','Sushi','Fitness','Coffee','Tacos'}
week_starts = pd.date_range(start = "2020-02-10", freq='7D', end="2020-12-21")
common = "./data/sg/common_ids_till_week_2020-12-21.json"

def to_py_list(list_str):
    return ast.literal_eval(list_str)

def gen_hourly_time_series_for_a_week(start):
    pass


def load_id_hourly_vis_pickle_dict(date):
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/"

    input_file = generate_path(date) + "id_hourly_vis_time_series_dict.pickle"
    table = pickle.load(open(input_file, 'rb'))
    return table

def extract_city_data_from_global_data(start_date):
    """
    Extract City data from one week
    :param start_date: datetime.datetime object specifying the starting date of the week
    :return: None
    """

    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/"

    # print("Extracting", start_date.strftime("%Y-%m-%d"))
    output_name = generate_path(start_date) + "city_data.csv"
    # print("Output name: " + output_name)
    city_data = []
    for i in range(1,5):
        input_name = generate_path(start_date) + "patterns-part" + str(i) + ".csv"
        # print(input_name)
        data = pd.read_csv(input_name)
        df = pd.DataFrame(data)
        for ct in cities:
            one_city = df.loc[(df['city'].str.contains(ct)) & (df['iso_country_code'] == 'US') & (df['region'] == 'CA')& (~df.city.isin(exclude))]
            city_data.append(one_city)
    select_LA = pd.concat(city_data)
    select_LA.to_csv(output_name, encoding='utf-8')
    print(select_LA.size)




features = [
    'safegraph_place_id',
    'visits_by_each_hour',
    "date_range_start",
    "date_range_end"
]

def extract_useful_features_from_city_data(date):
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/"

    input_file = generate_path(date) + "city_data.csv"
    output_file = generate_path(date) + "hourly_vis_only.csv"
    total = pd.read_csv(input_file)
    useful = total[features]
    useful.to_csv(output_file)

def transform_hourly_vis_only_data_to_pickle_dict(date):
    # Transform .csv file to pickle dictionary
    # Keys are sgid
    # Values are panda series
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/"

    input_file = generate_path(date) + "hourly_vis_only.csv"
    output_file = generate_path(date) + "id_hourly_vis_time_series_dict.pickle"

    raw = pd.read_csv(input_file, index_col='safegraph_place_id')
    res = dict()
    for index, row in raw.iterrows():
        vals = to_py_list(row['visits_by_each_hour'])
        res[index] = vals

    with open(output_file, 'wb') as output:
        pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)


def extract_common_ids(date_range):
    def generate_pickle_file_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/id_hourly_vis_time_series_dict.pickle"

    def generate_common_id_file_path(end_date):
        return "./data/sg/common_ids_till_week_" + end_date.strftime('%Y-%m-%d') + ".json"
    res = None
    pbar = tqdm(desc="Extracting common location ids from the weekly data",total=len(date_range))
    for d in date_range:
        input = generate_pickle_file_path(d)
        id_set = set(pickle.load(open(input, 'rb')).keys())
        if res is None:
            res = id_set
        else:
            res = res.intersection(id_set)
        pbar.update()
    print("Finished! common id count = " + str(len(res)))
    output_name = generate_common_id_file_path(date_range[-1])
    with open(output_name, 'w') as f:
        json.dump(list(res), f)


def generate_complete_hourly_vis_cnt(date_range):
    def generate_common_id_file_path(end_date):
        return "./data/sg/common_ids_till_week_" + end_date.strftime('%Y-%m-%d') + ".json"
    def generate_complete_hourly_vis_file_path(end_date):
        return "./data/sg/all_hourly_vis_till_week_" + end_date.strftime('%Y-%m-%d') + ".json"
    def generate_pickle_file_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/id_hourly_vis_time_series_dict.pickle"
    common_id_file_name = generate_common_id_file_path(date_range[-1])
    common = json.load(open(common_id_file_name, 'r'))

    res = dict()
    for id in common:
        res[id] = []


    pbar = tqdm(total = len(date_range), desc = "Generating hourly visit count over entire date range for common locations")
    for d in date_range:
        input = generate_pickle_file_path(d)
        data_dict = pickle.load(open(input, 'rb'))
        for id in common:
            res[id] = res[id] + data_dict[id]
        gc.collect()
        pbar.update()

    output = generate_complete_hourly_vis_file_path(date_range[-1])
    with open(output, 'w') as f:
        json.dump(res, f)

def generate_location_address_look_up(date_range):
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/city_data.csv"
    def generate_id_address_look_up_file_index():
        return "./data/sg/address_lookup.json"

    pbar = tqdm(total=len(date_range), desc="Generating location id-address lookup")

    res = dict()

    for d in date_range:
        input_file = generate_path(d)
        df = pd.read_csv(input_file, index_col='safegraph_place_id')
        useful = df['street_address']
        for index, val in useful.items():
            if index in res:
                continue
            else:
                res[index] = val
        gc.collect()
        pbar.update()
    with open(generate_id_address_look_up_file_index(), 'w') as f:
        json.dump(res, f)


def generate_location_name_look_up(date_range):
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/city_data.csv"

    res = dict()
    pbar = tqdm(total=len(date_range), desc="Generating location id-name lookup")
    for d in date_range:
        input = generate_path(d)
        df = pd.read_csv(input, index_col='safegraph_place_id')
        useful = df['location_name']
        for index, val in useful.items():
            if index in res:
                continue
            else:
                res[index] = val
        gc.collect()
        pbar.update()
    with open(loc_names, 'w') as f:
        json.dump(res, f)


def check_common_id_coverage():
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/id_hourly_vis_time_series_dict.pickle"
    common_id = set(json.load(open(common, 'r')))
    for d in week_starts:
        input = generate_path(d)
        df = pickle.load(open(input, 'rb'))
        can = 0
        tt = 0
        for id, val in df.items():
            curr = np.sum(list(val))
            if id in common_id:
                can += curr
            tt += curr
        print(d)
        print("\t" + "Common ids can cover: {:.2f}".format(can / tt))


def check_filter_coverage():
    names = json.load(open(loc_names, 'r'))
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/id_hourly_vis_time_series_dict.pickle"

    tt_vis = 0
    can_do = 0
    pbar = tqdm(total=len(week_starts), desc="Counting: ")
    for d in week_starts:
        input = generate_path(d)
        df = pickle.load(open(input, 'rb'))
        for id, val in df.items():
            nm = names[id]
            curr = np.sum(list(val))
            can_filter = False
            for word in nm.split(' '):
                if word in filter:
                    can_filter = True
                    break
            if can_filter:
                can_do += curr
            tt_vis += curr
        pbar.update()
    print("\t" + "Fraction of visits that is likely to be clusterable by keywords: {:.2f}".format(can_do / tt_vis))

def generate_frequent_word_weighted_average():
    names = json.load(open(loc_names, 'r'))
    stats = dict()
    output = "./data/frequent_words_in_name.json"
    def generate_path(date):
        return "./data/sg/" + date.strftime('%Y-%m-%d') + "/id_hourly_vis_time_series_dict.pickle"


    pbar = tqdm(total = len(week_starts), desc = "Counting words")
    for d in week_starts:
        data = pickle.load(open(generate_path(d), 'rb'))
        for id, val in data.items():
            weight = np.sum(val)
            name = names[id]
            for word in name.split(' '):
                stats[word] = int(stats.get(word, 0) + weight)
        pbar.update()
        gc.collect()
    stats = dict(sorted(stats.items(), key=operator.itemgetter(1),reverse=True))


    with open(output, 'w') as f:
        json.dump(stats,f)





#
#
# gen_city = pd.date_range(start = '2020-08-24', freq='7D', end="2020-12-21")
# gen_pickle = pd.date_range(start = '2020-02-10', freq='7D', periods=46)
#
# pbar = tqdm(total= len(gen_city))
# for d in gen_city:
#     extract_useful_features_from_city_data(d)
#     pbar.update()
#
# for d in gen_pickle:
#     transform_hourly_vis_only_data_to_pickle_dict(d)
#
#

# total = pd.date_range(start = '2020-02-10', freq='7D', end="2020-12-21")
# generate_location_address_look_up(total)
# extract_common_ids(total)
# check_filter_coverage()
# check_common_id_coverage()
# generate_location_name_look_up(week_starts)
# generate_frequent_word_weighted_average()



