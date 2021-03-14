import numpy as np
import pandas as pd
from SI import SI




def get_cumulative_infection(filename = "./data/LA_County_cumulative_infection.csv"):
    data = pd.read_csv(filename, parse_dates=['date_use'])
    res = pd.Series(data['cumulative_tests_pos'].values, index=data['date_use'])
    # Note that data here is sorted descend
    res = res[3:]
    return res


def get_la_google_mobility_from_national_google_mobility(filename = "./data/m2/2020_US_Region_Mobility_Report.csv"):
    data = pd.read_csv(filename)
    res = data[data['sub_region_2'] == 'Los Angeles County']
    res.to_csv("la_county_mobility.csv")
    return res


def load_google_mobility():
    google_mobility_score = pd.read_csv("./data/m2/la_county_mobility.csv", parse_dates=["date"])
    cols = google_mobility_score.columns[len(google_mobility_score.columns) - 6:len(google_mobility_score.columns)]  # columns we want
    newnames = {}
    for c_name in cols:
        newnames[c_name] = c_name.split('_')[0]
    google_mobility_score = google_mobility_score.rename(columns=newnames)
    return google_mobility_score


def calculate_real_google_mobility_score(data, gamma):
    cols = ['retail', 'grocery', 'parks', 'transit', 'workplaces']
    for c in cols:
        data[c] = (100 + data[c]) ** gamma


