import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

class SI:
    '''
    A generalized framework for Infection Rate SI-based models.
    Author: Kangmin Tan

    Sample Usage:
        model = SI(daily_cumulative_infection_pandas_time_series)
        model.add_mobility_term(daily_mobility_data_pandas_time_series_1)
        model.add_mobility_term(daily_mobility_data_pandas_time_series_2)
        .
        .
        .
        model.add_mobility_term(daily_mobility_data_pandas_time_series_i)
        .
        .
        .
        model.learn_betas()
        model.forecast(14, draw_plot = True)
    '''
    __version__ = '1.0'

    def __init__(self, cumulative_infection, smoothing = 7, k = 2, jp = 5, alpha = 0.95, lag = 1, split = 0.9, population = 10040000, split_date = None):
        """
        [...jp days...] [...jp days...] [...lag...] current_day
        | -------- k windows ---------|

        :param cumulative_infection: Pandas time Series, daily data
        :param k: the number of windows
        :param jp: size of each window
        :param alpha: is not used in this version of the implementation
        :param split: ratio of training data set
        :param smoothing: window size to smooth the data on
        """
        self.cumulative_infection = cumulative_infection.sort_index()
        # Hyper parameters
        self.k = k
        self.jp = jp
        self.alpha = alpha
        self.split = split
        self.lag = lag
        assert(lag > 0)
        self.mobility = []
        self.mobility_literals = []
        assert(smoothing % 2 == 1)
        self.smoothing = smoothing
        self.__prep_delta_infection()
        self.locked = False
        self.population = population
        self.split_date = split_date


    def add_mobility_term(self, data, literal):
        """
        This function adds a mobility term to the model

        In learn_betas(), the model would use the mobility data on day i to learn for day i.
        The user is responsible for pre-processing data so that the data on day i is the right one to be used in learning
        .
        :param data: Should be Pandas time series, daily data
        :param literal: the name of the dimension, as a string
        :return: None
        """
        assert(not self.locked)
        self.mobility.append(data)
        self.mobility_literals.append(literal)


    def __calculate_start_and_end_time(self):
        """
        Based on jp, k, and the starting time of mobility dimensions, the available range of time series data may vary.
        This private function pre-calculates a bound (self.available_date_range) to avoid out-of-bound issues in learning and forecasting.
        :return: None
        """

        start_time = self.delta_infection.index[self.jp * self.k + self.lag]
        for data in self.mobility:
            # Get the latest start time out of all the mobility dimensions
            start_time = max(start_time, data.index[0])

        end_time = self.delta_infection.index[-1]
        for data in self.mobility:
            # Get the earliest end time out of all the mobility dimensions
            end_time = min(end_time, data.index[-1])

        self.available_date_range = pd.date_range(start = start_time, end = end_time)

        self.__calculate_split()

    @staticmethod
    def kz_smooth(series, window, iterations=1):
        """KZ filter implementation
        series is a pandas series
        window is the filter window m in the units of the data (m = 2q+1)
        iterations is the number of times the moving average is evaluated
        """
        # Window size must be odd
        assert(window % 2 == 1)
        z = series.copy()
        for i in range(iterations):
            z = z.rolling(window=window, min_periods=1, center=True).mean()
        return z

    def __calculate_split(self):
        """
        Calculate training / testing split point based on available data range
        :return: None
        """

        if not self.split_date:
            split_idx = int(len(self.available_date_range) * self.split)
            self.split_date = self.available_date_range[split_idx]
            self.training_date_range = self.available_date_range[:split_idx]
            self.test_date_range = self.available_date_range[split_idx:]
        else:
            index = -1
            print(self.available_date_range, self.split_date)
            for i in range(len(self.available_date_range)):
                if self.available_date_range[i] == self.split_date:
                    index = i

            if index == -1:
                print("Available Date Range:")
                print("\t" + str(self.available_date_range))

                raise Exception("Specified split date not in available date range. This is probably caused by insufficient data")
            self.training_date_range = self.available_date_range[:index]
            self.test_date_range = self.available_date_range[index:]

        print("Data Split:")
        print("\tTraining data: " + str(len(self.training_date_range)) + "days")
        print("\tTest data: " + str(len(self.test_date_range)) + "days")
        print("\tSplitting point: " + str(self.split_date))

    def __prep_delta_infection(self):
        """
        Prepare the delta infection
        :return: None
        """
        # The first entry would always be NaN so take out one
        raw_delta_infection = self.cumulative_infection.diff()[1:]

        # Smooth delta infection based on the given smoothing window
        self.delta_infection = self.kz_smooth(raw_delta_infection, self.smoothing)



    def learn_betas(self):
        # User should call this method after adding all necessary mobility dimensions
        # No more mobility dimension should be added after calling this method
        self.locked = True

        self.__calculate_start_and_end_time()

        # Should be very easy to build the X and Y here, since we have all the boundary calculated
        # Y should just be the delta on that day
        # there are k number of columns in X corresponding to delta I
        # Since the responsibility of pre-processing mobility dimension data is on the user, we can simply use the values in time series

        # Start assembling X and Y
        # First k columns in X are non-mobility terms, then the mobility terms follows
        # Rows of x are the dates in self.trainable_date_range, the range precalculated above to avoid date out of bound
        X = np.zeros((len(self.training_date_range), int(self.k) + len(self.mobility)))

        # Start with the first k columns in X
        for i in range(self.k):
            for j in range(len(self.training_date_range)):
                day = self.training_date_range[j]

                # Shift lag days ahead
                day = day - pd.DateOffset(self.lag)

                # Shift i*jp days ahead
                day = day - pd.DateOffset(i * self.jp)


                # Get new infection over jp days
                begin_date = day - pd.DateOffset(self.jp)
                end_date = day

                delta_I = self.cumulative_infection[end_date] - self.cumulative_infection[begin_date]
                assert(delta_I >= 0)

                X[j,i] = delta_I

        # Fill in mobility data in X
        for m in range(len(self.mobility)):
            for j in range(len(self.training_date_range)):
                day = self.training_date_range[j]
                X[j,m+self.k] = self.mobility[m][day]

        # Multiply by susceptible population
        for j in range(len(self.training_date_range)):
            day = self.training_date_range[j] - pd.DateOffset(self.lag)
            susceptible_population = (self.population - self.cumulative_infection[day])
            for i in range(X.shape[1]):
                X[j,i] *= susceptible_population


        # Construct Y
        Y = np.zeros((len(self.training_date_range),))
        for j in range(len(self.training_date_range)):
            day = self.training_date_range[j]
            Y[j] = self.delta_infection[day]
        # print(X)
        # print(Y)

        # Train
        self.betas = lsq_linear(X, Y, bounds=(0, np.inf)).x


        # Show betas for each term
        print("Training finished, betas:")
        for i in range(self.k):
            print("Delta infection term k="+str(i) + " : " + str(self.betas[i]))
        for i in range(len(self.mobility)):
            print("Mobility term " + self.mobility_literals[i] + ": " + str(self.betas[self.k + i]))

    def forecast(self, forward, make_plot = False):
        """

        :param forward:
        :param make_plot: if True will draw a plot
        :return:
        """
        num_of_days = forward

        # If there are mobility dimensions, have to make sure there's enough data
        if(len(self.mobility) and forward > len(self.test_date_range)):
            print("Warning! Not enough data, days to forecast changed to " + str(len(self.test_date_range)))
            num_of_days = len(self.test_date_range)

        # split_date is the first day that we didn't consider in self.learn_betas()
        start_date = self.split_date
        end_date =  start_date + pd.DateOffset(num_of_days-1)

        # Initialize predictions to all zeros
        predictions = pd.Series(np.zeros(num_of_days), index=pd.date_range(start=start_date, end=end_date))

        # Make a copy of the cumulative infection since we are going to change it on the way of prediction
        dynamic_cumu_infection = self.cumulative_infection.copy()

        # Make predictions
        for day in predictions.index:

            # Find susceptible population
            day_ptr = day - pd.DateOffset(self.lag)
            susceptible_population = self.population - dynamic_cumu_infection[day_ptr]

            d_I = 0
            # Get the contribution of the mobility terms
            for i in range(len(self.mobility)):
                curr_beta = self.betas[self.k + i]
                d_I += susceptible_population * curr_beta *  self.mobility[i][day]

            # Calculate the contribution of k SI model terms
            # Note that the k windows goes from later to earlier
            for i in range(self.k):
                curr_beta = self.betas[i]
                day_ptr = day - pd.DateOffset(self.lag)
                day_ptr -= pd.DateOffset(self.jp * i)

                # Get new infection over jp days
                begin_date = day_ptr - pd.DateOffset(self.jp)
                end_date = day_ptr

                infections = dynamic_cumu_infection[end_date] - dynamic_cumu_infection[begin_date]
                d_I += (infections * susceptible_population * curr_beta)

            predictions[day] = d_I

            yesterday = day - pd.DateOffset(1)
            dynamic_cumu_infection[day] = dynamic_cumu_infection[yesterday] + d_I


        mae, day_cnt = self.MAE(self.delta_infection, predictions)
        self.predictions = predictions

        if make_plot:
            plt.grid(True)
            plt.plot(self.delta_infection, 'ro')
            plt.plot(self.delta_infection, 'r-')
            plt.plot(predictions, 'yo')
            plt.plot(predictions, 'y-')
            y_min,y_max = plt.gca().get_ylim()
            y_scale = np.linspace(y_min,y_max)
            plt.text(self.delta_infection.index[10],y_scale[int(len(y_scale)*(2/4))],"Mean Absolute Error: " + "{0:.3f}".format(mae), fontsize=10)
            plt.text(self.delta_infection.index[10], y_scale[int(len(y_scale)*(3/4))], "Comparable prediction: " + str(day_cnt) + " days", fontsize=10)
            plt.show()

        return mae, day_cnt

    @staticmethod
    def MAE(true, prediction):
        truth_arr = []
        prediction_arr = []
        for date in true.index:
            if(prediction.__contains__(date)):
                truth_arr.append(true[date])
                prediction_arr.append(prediction[date])
        return mean_absolute_error(truth_arr, prediction_arr),len(truth_arr)

    def last_week_absolute_percentage_error(self):
        assert(len(self.predictions.index) >=7)
        last_day = self.predictions.index[-1]
        first_day = last_day - pd.DateOffset(6)

        print("Calculate error date range:")
        print(first_day)
        print(last_day)

        drange = pd.date_range(start=first_day, end=last_day)

        pred_sm = 0
        truth = 0

        for d in drange:
            assert(d in self.delta_infection.index)
            truth += self.delta_infection[d]
            pred_sm += self.predictions[d]
        return abs(pred_sm - truth) / truth

    def last_week_absolute_error(self):
        assert (len(self.predictions.index) >= 7)
        last_day = self.predictions.index[-1]
        first_day = last_day - pd.DateOffset(6)

        print("Calculate error date range:")
        print(first_day)
        print(last_day)

        drange = pd.date_range(start=first_day, end=last_day)

        pred_sm = 0
        truth = 0

        for d in drange:
            assert (d in self.delta_infection.index)
            truth += self.delta_infection[d]
            pred_sm += self.predictions[d]

        return abs(pred_sm - truth)