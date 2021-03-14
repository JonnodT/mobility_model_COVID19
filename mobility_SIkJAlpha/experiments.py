import pickle
from SI import *
from SI_convo import SI_m
import json
from tqdm import tqdm
from general_data_preprocess import get_cumulative_infection, load_google_mobility,calculate_real_google_mobility_score
import matplotlib.pyplot as plt



cumu = get_cumulative_infection()

# Load and smooth google mobility
google_mobility = load_google_mobility()
calculate_real_google_mobility_score(google_mobility, gamma=4)
loc_names = "./data/sg/location_name_lookup.json"
all_cols = ['retail', 'grocery', 'parks', 'transit', 'workplaces']
week_starts = pd.date_range(start = "2020-02-10", freq='7D', end="2020-12-21")
day_starts = pd.date_range(start = "2020-02-10T00:00:00", freq='1D', end="2020-12-27T00:00:00")
cluster_mobility = "./data/sg/clustering/cluster_mobility.pickle"


MANUAL_CLST_DIMENSION_FILE = "./data/sg/manual_clustering/dimension.json"
AUTO_CLST_DIMENSION_FILE = "./data/sg/clustering/cluster_mobility.pickle"
MANUAL_CLST = ["dining", "school", "shopping", "hotel", "recreation", "airport"]

words = {'Park', 'Restaurant', 'School','Church', 'Bar', 'Grill', 'Pizza', 'Bakery','Donut','Seafood','Library','Museum', 'Burgers', 'Inn', 'Pharmacy','Hotel','Sushi','Fitness','Coffee','Tacos'}


def tune_googe():
    # ()
    k_range = range(1,3)
    jp_range = range(2,6)
    lag_range = range(1,5)
    gamma_range = range(2,6)
    TEST_RANGE = pd.date_range(start='2020-09-06', end="2020-11-08", freq="7D")
    google_choice = ['workplaces', 'retail']
    TEST_FORWARD = 21
    curr_err = 1000000
    curr_para = (0,0,0,0)

    def run_on_google_model(date,k,jp,lag, mobility):
        model = SI_m(cumu, split_date=date)
        def generate_dimension(name, mobility_data):
            return pd.Series(mobility_data[name].values, index=mobility_data['date'])

        dimensions = {}
        for name in google_choice:
            dimensions[name] = generate_dimension(name, google_mobility)
            assert (name in google_mobility.columns)
        for dm in dimensions:
            model.add_mobility_term(dimensions[dm], dm, k, jp, lag)
        model.learn_betas()

        res = model.forecast(TEST_FORWARD)
        assert (res[1] == TEST_FORWARD)
        return model.last_week_absolute_percentage_error()

    for k in k_range:
        for jp in jp_range:
            for lag in lag_range:
                for gamma in gamma_range:
                    curr = (k, jp, lag, gamma)
                    google_mobility = load_google_mobility()
                    calculate_real_google_mobility_score(google_mobility, gamma=gamma)
                    total_err = 0
                    for d in TEST_RANGE:
                        err = run_on_google_model(date=d, jp=jp, k=k, lag=lag,mobility=google_mobility)
                        total_err += err
                    if total_err < curr_err:
                        curr_err = total_err
                        curr_para = curr
    print(curr_para)


def compare_all_methods_error_over_time(days_forward, ax):
    TEST_RANGE = pd.date_range(start='2020-08-23', end="2020-11-15", freq="7D")
    # TEST_RANGE = pd.date_range(start='2020-08-23', end="2020-09-25", freq="7D")
    TEST_FORWARD = days_forward
    # plt.figure(figsize=(15, 7))
    manual_clst = json.load(open(MANUAL_CLST_DIMENSION_FILE,'r'))
    auto_clst = pickle.load(open(AUTO_CLST_DIMENSION_FILE,'rb'))
    google_mobility = load_google_mobility()
    calculate_real_google_mobility_score(google_mobility, gamma=3)

    AUTO_CLUSTER_CNT = len(auto_clst.keys())


    google_choice = ['workplaces', 'retail']
    manual_cluster_choice = ['recreation', 'shopping', 'airport']

    for clst in MANUAL_CLST:
        assert(len(day_starts) == len(manual_clst[clst]))
        manual_clst[clst] = pd.Series(manual_clst[clst], index=day_starts)

    ks = {'Generic Model': 2, 'Google Mobility': 2, 'Hierarchical Clustering': 2, 'Manual Clustering': 2}
    jps = {'Generic Model': 7, 'Google Mobility': 7, 'Hierarchical Clustering': 7, 'Manual Clustering': 7}
    lags = {'Generic Model':2, 'Google Mobility':2, 'Hierarchical Clustering':2, 'Manual Clustering':2}

    colors = {'No Mobility': 'k', 'Google Mobility': 'c', 'Hierarchical Clustering': 'r', 'Manual Clustering': 'b'}

    def run_on_generic_model(date):
        k = ks['Generic Model']
        jp = jps['Generic Model']
        lag = lags['Generic Model']
        model = SI(cumu, k=k, jp=jp, lag=lag, split_date=date)
        model.learn_betas()
        res = model.forecast(TEST_FORWARD)
        assert(res[1] == TEST_FORWARD)
        return model.last_week_absolute_error()


    def run_on_google_model(date):

        k = ks['Google Mobility']
        jp = jps['Google Mobility']
        lag = lags['Google Mobility']

        model = SI_m(cumu, split_date=date)

        def generate_dimension(name, mobility_data):
            return pd.Series(mobility_data[name].values, index=mobility_data['date'])

        dimensions = {}
        for name in google_choice:
            dimensions[name] = generate_dimension(name, google_mobility)
            assert(name in google_mobility.columns)
        for dm in dimensions:
            model.add_mobility_term(dimensions[dm], dm, k, jp, lag)
        model.learn_betas()

        res = model.forecast(TEST_FORWARD)
        assert(res[1] == TEST_FORWARD)
        return model.last_week_absolute_error()

    def run_on_auto_clst_model(date):
        k = ks['Hierarchical Clustering']
        jp = jps['Hierarchical Clustering']
        lag = lags['Hierarchical Clustering']
        model = SI_m(cumu, split_date=date)
        for cls in range(AUTO_CLUSTER_CNT):
            model.add_mobility_term(auto_clst[cls], 'Class' + str(cls),k=k, jp=jp, lag=lag)
        model.learn_betas()
        res = model.forecast(TEST_FORWARD)
        assert (res[1] == TEST_FORWARD)
        return model.last_week_absolute_error()

    def run_on_manual_clst_model(date):
        k = ks['Manual Clustering']
        jp = jps['Manual Clustering']
        lag = lags['Manual Clustering']
        model = SI_m(cumu, split_date=date)
        for clst in manual_cluster_choice:
            assert(clst in manual_clst.keys())
            model.add_mobility_term(manual_clst[clst], clst, k=k, jp=jp, lag=lag)
        model.learn_betas()
        res = model.forecast(TEST_FORWARD)
        assert (res[1] == TEST_FORWARD)
        return model.last_week_absolute_error()


    errors = {'No Mobility':[], 'Google Mobility':[], 'Hierarchical Clustering':[], 'Manual Clustering':[]}


    for day in TEST_RANGE:
        errors['No Mobility'].append(run_on_generic_model(day))
        errors['Google Mobility'].append(run_on_google_model(day))
        errors['Hierarchical Clustering'].append(run_on_auto_clst_model(day))
        errors['Manual Clustering'].append(run_on_manual_clst_model(day))

    for model_name in errors.keys():
        errors[model_name] = pd.Series(errors[model_name], index=TEST_RANGE)


    for model_name, errs in errors.items():
        ax.plot(errs, colors[model_name], linestyle="dashed", marker='o',label=model_name)
    # plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])
    ax.set_title("" + str(int(TEST_FORWARD/7)) + " Week ahead forecasts", fontsize=17)
    # ax.grid(True)
    # ax.legend(loc=0, fontsize = 28)
    # ax.set_yticks(fontsize = 15)
    ax.tick_params(labelsize=14)
    sparse_tick = TEST_RANGE[0::2]
    ax.set_xticks(sparse_tick)


def test_manual_clustering_error_over_time():
    def square(a):
        # n * (n-1) / 2
        b = a - 1
        b[b < 0] = 0
        return (a * b) / 2

    clusters = json.load(open(MANUAL_CLST_DIMENSION_FILE, 'r'))

    for c in MANUAL_CLST:
        clusters[c] = pd.Series(clusters[c], index=day_starts)



    selected_clusters = ['shopping', 'recreation', 'airport']
    try_range = pd.date_range(start='2020-09-10', end="2020-10-10")
    forward = 60

    errs = []
    for d in try_range:
        model = SI_m(cumu, split_date=d)
        for c in selected_clusters:
            model.add_mobility_term(clusters[c],c,2,5,2)
        model.learn_betas()
        err = model.forecast(forward)
        errs.append(err[0])


    gerrs =[]
    for d in try_range:
        gmodel = SI(cumu,k=2,jp=5,lag=2, split_date=d)
        gmodel.learn_betas()
        gmodel.forecast(forward)
        gerrs.append(err[0])

    all_err_generic = pd.Series(gerrs, index=try_range)
    all_err_manual = pd.Series(errs, index=try_range)
    plt.plot(all_err_manual, label="manual cluster model MAE")
    plt.plot(all_err_generic, label="generic model MAE")
    plt.title("MAE" + " days=" + str(forward))
    plt.legend(loc=0)

    plt.show()


def manual_clusters_mode_alone():
    def square(a):
        # n * (n-1) / 2
        b = a - 1
        b[b < 0] = 0
        return (a * b) / 2

    clusters = json.load(open(MANUAL_CLST_DIMENSION_FILE, 'r'))

    for c in MANUAL_CLST:
        clusters[c] = pd.Series(clusters[c], index=day_starts)

    model = SI_m(cumu, split_date=pd.to_datetime('2020-11-10'))

    selected_clusters = ['shopping','recreation','airport','school']

    for c in selected_clusters:
        if c not in MANUAL_CLST:
            assert(0)
        else:
            model.add_mobility_term(clusters[c],c,3,6,3)
            plt.plot(clusters[c], label=c)

    plt.legend(loc=0)
    plt.show()

    model.learn_betas()
    model.forecast(20, make_plot=True)
    print(model.last_week_absolute_percentage_error())








def compute_lag_sliding_window(data, lag, window_size):
    """
    Data should be panda series with date as index
    :param data:
    :param lag:
    :param window_size:
    :return: Processed pandas series
    """
    new_index = data.index[lag+window_size:]
    print(new_index)
    nlen = len(new_index)
    ans = pd.Series(np.zeros(nlen), index=new_index)
    for day in new_index:
        curr = 0
        for i in range(window_size):
            curr += data.shift(lag + i)[day]
        ans[day] = curr / window_size

    return ans







def use_safegraph_data_as_a_single_dimension():
    """
    method 1 (m1)
    :return:
    """
    def load_mobility_data():
        return pickle.load( open( "./data/m1/daily_total_vis_cnt_pd_series.pickle", "rb" ) )


    def extract_mobility_dimension():
        raw_dict = load_mobility_data()
        print("Load finished")
        mobility_dict = dict()

        pbar = tqdm(total=len(raw_dict))
        for k,curr in raw_dict.items():
            for start_day, arr in curr.items():
                date = pd.to_datetime(start_day)
                # TODO: Figure out a way to build create an empty serie and insert values to it!!!
                mobility_dict[date] = mobility_dict.get(date,0) + np.sum(arr)
            pbar.update(1)
        pbar.close()
        return mobility_dict



    mob = load_mobility_data()

    # plt.plot(mob)
    # plt.show()

    model = SI(cumu, k=2, jp=5, lag=1, split_date='2020-08-05')
    # model.add_mobility_term(compute_lag_sliding_window(mob, lag=i, window_size=j), 'safegraph_mobility')
    model.learn_betas()
    model.forecast(12, make_plot=True)


    res = np.zeros((14,14))

    for i in range(1,14):
        for j in range(1,14):
            model = SI(cumu, k=2, jp=5, lag=1, split_date='2020-08-05')
            model.add_mobility_term(compute_lag_sliding_window(mob, lag=i, window_size=j), 'safegraph_mobility')
            model.learn_betas()
            mae,day = model.forecast(12)
            assert(day == 12)
            res[i][j] = mae

    for i in range(1,14):
        for j in range(1,14):
            print("Lag = "+ str(i)+". window_size = " + str(j) + " MAE: "+ str(res[i][j]))


    # mob = load_mobility_data()
    # start_date = None
    # end_date = None
    # for k in mob:
    #     day = pd.to_datetime(k)
    #     if not start_date:
    #         start_date = day
    #     if not end_date:
    #         end_date = day
    #     start_date = min(start_date, day)
    #     end_date = max(end_date,day)
    # range = pd.date_range(start = start_date, end = end_date)
    # data = pd.Series(np.zeros(len(range)), index = range)
    # for k in mob:
    #     day = pd.to_datetime(k)
    #     data[day] = mob[k]
    # with open("./data/m1/daily_total_vis_cnt_pd_series.pickle", 'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    #



def use_google_mobility_as_separate_dimensions():
    """
    Method 2 (m2)
    :return:
    """
    # cols = ['retail', 'grocery', 'parks', 'transit', 'workplaces']
    cols = ['workplaces', 'retail']


    def generate_dimension(name,mobility_data):
        return pd.Series(mobility_data[name].values, index=mobility_data['date'])

    dimensions = {}
    for name in cols:
        dimensions[name] = generate_dimension(name, google_mobility)
    model = SI(cumu, k=2, jp=7, lag=1, split_date='2020-11-13')

    for d in dimensions:
        model.add_mobility_term(compute_lag_sliding_window(dimensions[d], lag=5, window_size=30), d)
    print(cumu)
    model.learn_betas()
    model.forecast(17)


    day_cnt = 7
    try_range = pd.date_range(start='2020-10-10', end="2020-11-01")
    no_mobility = []
    with_mobility = []
    for d in try_range:
        model = SI(cumu, k=2, jp=7, lag=1, split_date=d)
        model.learn_betas()
        mae, day = model.forecast(day_cnt)
        assert(day == day_cnt)
        no_mobility.append(mae)

        model2 = SI(cumu, k=2, jp=7, lag=1, split_date=d)
        for d in dimensions:
            model2.add_mobility_term(compute_lag_sliding_window(dimensions[d], lag=7, window_size=30), d)
        model2.learn_betas()
        mae, day = model2.forecast(day_cnt)
        assert (day == day_cnt)
        with_mobility.append(mae)
    plt.grid(True)
    plt.plot(with_mobility, 'ro')
    plt.plot(with_mobility, 'r-')
    plt.plot(no_mobility, 'ko')
    plt.plot(no_mobility, 'k-')
    plt.show()

def use_clustering_data_as_dimensions():
    pass
    # TODO: 补全data，在这做clustering


def test_convolution_SI():

    drange = pd.date_range(start='2020-10-10', end='2020-11-01')
    forward = 30
    no_mobility = []
    with_mobility = []
    for d in drange:

        model = SI_m(cumu, split_date=d)
        def generate_dimension(name,mobility_data):
            return pd.Series(mobility_data[name].values, index=mobility_data['date'])
        cols = ['workplaces','retail','transit']
        dimensions = {}
        for name in cols:
            dimensions[name] = generate_dimension(name, google_mobility)
        for dm in dimensions:
            model.add_mobility_term(dimensions[dm], dm, 1, 10, 1)
        model.learn_betas()
        mae, day = model.forecast(forward)
        assert(day == forward)
        with_mobility.append(mae)

        print(d)
        old = SI(cumu,k=1,jp=10,lag=1,split_date=d)

        old.learn_betas()
        mae, day = old.forecast(forward)
        assert(day == forward)
        no_mobility.append(mae)
    print(with_mobility,no_mobility)
    plt.grid(True)
    plt.plot(with_mobility, 'ro')
    plt.plot(with_mobility, 'r-')
    plt.plot(no_mobility, 'ko')
    plt.plot(no_mobility, 'k-')
    plt.show()

def convolution_SI_alone():
    d = '2020-11-04'
    forward = 20
    model = SI_m(cumu, split_date=d)

    def generate_dimension(name, mobility_data):
        return pd.Series(mobility_data[name].values, index=mobility_data['date'])

    cols = ['workplaces', 'retail', 'transit']
    dimensions = {}
    for name in cols:
        dimensions[name] = generate_dimension(name, google_mobility)
    for dm in dimensions:
        model.add_mobility_term(dimensions[dm], dm, 1, 10, 1)
    model.learn_betas()
    mae, day = model.forecast(forward, make_plot=True)
    assert (day == forward)


    bk = -1
    bjp = -1
    blag = -1
    curr = 30000000
    cols = ['workplaces', 'retail', 'transit']
    #
    # def generate_dimension(name, mobility_data):
    #     return pd.Series(mobility_data[name].values, index=mobility_data['date'])
    # for k in range(1,4):
    #     for jp in range(7,22):
    #         for lag in range(1,13):
    #             model = SI_m(cumu, split_date=d)
    #             dimensions = {}
    #             for name in cols:
    #                 dimensions[name] = generate_dimension(name, mobility)
    #             for dm in dimensions:
    #                 model.add_mobility_term(dimensions[dm], dm, k, jp, lag)
    #             model.learn_betas()
    #             mae, day = model.forecast(forward)
    #             if mae < curr:
    #                 bk = k
    #                 bjp = jp
    #                 blag = lag
    #                 curr =  mae
    #             assert (day == forward)
    #
    # print(bk)
    # print(bjp)
    # print(blag)
    # print(curr)

def find_best_parameter():
    bk = [-1,-1,-1,-1]
    bjp = [-1,-1,-1,-1]
    blag = [-1,-1,-1,-1]
    curr = 30000000
    cols = ['workplaces', 'retail', 'transit','grocery']



def compare_three_models():
    """
    Experiment to test the result of
        1. No mobility feature
        2. Google Mobility
        3. Hierarchical clustering
    :return:
    """

    k = 2
    jp = 5
    lag = 1
    sundays = pd.date_range(start = "2020-05-10", end = "2020-11-08", freq="7D")
    # sundays = pd.date_range(start='2020-10-10', end='2020-11-01')
    # Selected columns of google mobility
    cols = ['workplaces', 'retail']
    # Read

    def read_cluster():
        dimensions = pickle.load(open(cluster_mobility, 'rb'))
        return dimensions
    clustering = read_cluster()
    # Number of clusters
    cluster_n = len(clustering)


    def generate_dimension(name, mobility_data):
        return pd.Series(mobility_data[name].values, index=mobility_data['date'])
    def prep_no_mobility(split):
        model = SI(cumu, split_date=split, k= k, jp = jp, lag = lag)
        model.learn_betas()
        return model

    def prep_google_model(split):
        model = SI_m(cumu, split_date=split)
        dimensions = {}
        for name in cols:
            dimensions[name] = generate_dimension(name, google_mobility)
        for dm in dimensions:
            model.add_mobility_term(dimensions[dm], dm, k, jp, lag)
        model.learn_betas()
        return model

    def prep_cluster_model(split):
        model = SI_m(cumu, split_date=split)
        for i in range(cluster_n):
            model.add_mobility_term(clustering[i], 'Cluster ' + str(i), k, jp, 5)
        model.learn_betas()
        return model

    x = 50
    nothing_res = []
    google_res = []
    clst_res = []
    for d in sundays:
        nothing = prep_no_mobility(d)
        google = prep_google_model(d)
        clst = prep_cluster_model(d)

        mae_nothing, day_nothing = nothing.forecast(x)
        assert(day_nothing == x)
        nothing_res.append(mae_nothing)

        mae_google, day_google = google.forecast(x)
        assert(day_google == x)
        google_res.append(mae_google)

        mae_clst, day_clst = clst.forecast(x)
        assert(day_clst == x)
        clst_res.append(mae_clst)
    plt.title("MAE" + " days=" + str(x))
    plt.plot(pd.Series(nothing_res, index=sundays), "ro")
    plt.plot(pd.Series(nothing_res, index=sundays), "r-", label="No Mobility")
    plt.plot(pd.Series(google_res, index=sundays), "bo")
    plt.plot(pd.Series(google_res, index=sundays), "b-", label="Google")
    plt.plot(pd.Series(clst_res, index=sundays), "ko")
    plt.plot(pd.Series(clst_res, index=sundays), "k-", label="cluster")
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()

def cluster_model_alone():

    d = pd.to_datetime('2020-08-25')
    forward = 100
    k = 2
    jp = 7
    lag = 2

    def read_cluster():
        dimensions = pickle.load(open(cluster_mobility, 'rb'))
        return dimensions
    clustering = read_cluster()
    # Number of clusters
    cluster_n = len(clustering)
    def prep_cluster_model(split):
        model = SI_m(cumu, split_date=split)
        for i in range(cluster_n):
            model.add_mobility_term(clustering[i], 'Cluster ' + str(i), k, jp, lag)
        model.learn_betas()
        return model


    model = prep_cluster_model(d)
    model.forecast(forward, make_plot=True)

def primitive_model_alone():
    d = pd.to_datetime('2020-07-13')
    forward = 35
    k = 2
    jp = 7
    lag=1
    model = SI(cumu, k=k, jp = jp, lag = lag, split_date=d)
    model.learn_betas()
    model.forecast(forward, make_plot=True)


def google_mobility_model_alone():
    k = 2
    jp = 7
    lag = 1
    forward = 30
    # sundays = pd.date_range(start='2020-10-10', end='2020-11-01')
    # Selected columns of google mobility
    cols = ['retail', 'workplaces']
    d = pd.to_datetime('2020-12-07')
    def generate_dimension(name, mobility_data):
        return pd.Series(mobility_data[name].values, index=mobility_data['date'])
    def prep_google_model(split):
        model = SI_m(cumu, split_date=split)
        dimensions = {}
        for name in cols:
            dimensions[name] = generate_dimension(name, google_mobility)
        for dm in dimensions:
            model.add_mobility_term(dimensions[dm], dm, k, jp, lag)
        model.learn_betas()
        return model
    model = prep_google_model(d)
    model.forecast(forward, make_plot=True)

def plot_3456_week_prediction_plot():
    fig, axes = plt.subplots(3,1,sharex=True, sharey=True, figsize=(10,6))
    compare_all_methods_error_over_time(21,axes[0])
    compare_all_methods_error_over_time(28,axes[1])
    compare_all_methods_error_over_time(35,axes[2])
    plt.xlabel('Forecast Days', fontsize=18)
    plt.ylabel('Absolute Error', fontsize=18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=14)
    # fig.text(.05, .5, 'Absolute Error', ha='center', va='center', rotation='vertical', fontsize=19)
    axes[0].legend(loc="upper left", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout(pad=1)
    plt.show()



# compare_three_models()
# cluster_model_alone()
# primitive_model_alone()
# google_mobility_model_alone()

# manual_clusters_mode_alone()
# test_manual_clustering_error_over_time()
# compare_all_methods_error_over_time()
# plot_3456_week_prediction_plot()
# tune_googe()
