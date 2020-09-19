from typing import List
import functools
from threading import Thread
from functools import partial
from contextlib import contextmanager
import itertools as it
import warnings
import signal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hyperopt import fmin, tpe, space_eval, hp
from hyperopt.pyll import scope
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA

import random
import os

from subprocess import check_output
from pathlib import Path
"""Starting variables for the models including the stock information files"""
WINDOW = 50
STDEV = 2.5
STARTING_FUNDS = 1000
FILENAMES = list(Path(r"C:\\Users\\i_miz\\Documents\\Visual_Studio_Projects\\Streaming_uploads\\Random_Streaming_Files\\Data_analytics\\Stock_Market_data\\Data\\Stocks").glob('*.txt'))

PRINTED = set()
def printonce(msg):
    if msg not in PRINTED:
        print(msg)
        PRINTED.add(msg)


class TimeoutException(Exception):
    pass


def timeout(timeout):
    """probably chinese virus idk.
    might be Italian Spaghetti. Don't know Italian so I couldn't say"""
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """more chinese virus words"""
            result = TimeoutException('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))
            def newFunc():
                try:
                    result = func(*args, **kwargs)
                except TimeoutException as e:
                    result = e
            t = Thread(target=newFunc)
            """Satan lives here"""
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except TimeoutException as je:
                print('error starting thread')
                raise je
            if isinstance(result, BaseException):
                raise result
            return result
        return wrapper
    return deco

"""
#REF: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands#:~:text=Bollinger%20Bands%20are%20envelopes%20plotted,Period%20and%20Standard%20Deviations%2C%20StdDev.
Short term: 10 day moving average, bands at 1.5 standard deviations. (1.5 times the standard dev. +/- the SMA)
Medium term: 20 day moving average, bands at 2 standard deviations.
Long term: 50 day moving average, bands at 2.5 standard deviations.
"""

def train_test_split(df, split_date):
    """Splits dates in between 2 sides of a set "Split Date Variable wihtin the dataframe"""
    return df[df["Date"] <= split_date], df[df["Date"] > split_date]

def simulator(price: List[float], buys: List[int], sells: List[int], pocket: float) -> float:
    """Takes buy/sell data and determines a total value gained or loss"""
    number_shares = 0
    buys = [(i, 'buy') for i in list(buys)]
    sells = [(i, 'sell') for i in list(sells)]
    buysells = buys + sells
    sbuysells = sorted(buysells)
    for (i, buysell) in sbuysells:
        if buysell == 'buy':
            if pocket and pocket > price[i]:
                number_shares += pocket // price[i]
                pocket = pocket % price[i]
        elif buysell == 'sell':
            pocket = price[i] * number_shares
            number_shares = 0
    pocket = price[-1] * number_shares
    return pocket

def model(series, ar, diff, mov_avg):
    """Attempts to use ARIMA model. If model works will print out (does this ever work?) if model
    doesn't work then will only give value of -1000 for all dates"""
    try:
        model = ARIMA(series, order=(int(ar),int(diff),int(mov_avg)))
        model_fit = model.fit(disp=0)
        out = model_fit.forecast()[0]
        printonce("Does this ever work?")
    except ValueError as e:
        return -1000
    return out

def get_files():
    """Finds the files used and places then into a pandas dataframe"""
    assert FILENAMES, "No filenames found"
    for f in FILENAMES[:10]:
        try:
            df = pd.read_csv(f)
        except:
            continue
        df["Avg"] = (df["High"] + df["Low"])/2
        yield df

@timeout(1)
def objective_parameter_tuning(parameters):
    """take the parameters and find the mean of all parameters by finding a range. Uses dataframe to
    get an average value for training. """
    ar, diff, mov_avg = int(parameters['ar']), int(parameters['diff']), int(parameters['mov_avg'])
    all_mse = []
    for df in get_files():
        train = df['Avg'].values[:int(len(df)*.66)]
        our_range = list(range(int(len(train)*.3), len(train)-2))
        #print("Here", our_range[0], our_range[-1])
        #print(type(train))
        pred = np.array([model(train[:int(i)], ar, diff, mov_avg) for i in our_range])
        #print("We Made it!")
        target = np.array([train[i] for i in our_range])
        if our_range:
            msa = ((pred - target) ** 2).mean()
            all_mse.append(msa)
    return np.mean(all_mse)

def hyperparameter_tuning():
    """take stock information and tune to determine best use parameters. Constant warnings will appear
    preventing you from completing this task so try except must be passed to time them out. If you 
    do not do this then you will get an infinite warning loop. """
    best_value, best_pars = float('inf'), dict()
    for (ar, diff, mov_avg) in tqdm(it.product(range(10), range(5), range(5)), desc="Hyperparameter Tuning", total=10*5*5):
        this_pars = {'ar': ar, 'diff': diff, 'mov_avg': mov_avg}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                this_value = objective_parameter_tuning(this_pars)
            except TimeoutException:
                continue
        if this_value < best_value:
            best_value = this_value
            best_pars = this_pars
            print(f"New Best Parameters: {best_pars} Value: {best_value}")
    print("Done!")
    if best_pars:
        return best_pars
    raise Exception("Didn't find any.")

def main():
    # parameters = hyperparameter_tuning()
    parameters = {'ar': 5, 'diff': 3, 'mov_avg': 0}
    model_ = partial(model, **parameters)
    avg_earnings = []
    for df in tqdm(get_files(), total=len(FILENAMES), leave=True, desc="Files"):
        test = df['Avg'].values[int(len(df)*.66):]
        our_range = list(range(int(len(test)*.3), len(test)))
        print("Predicting...")
        predictions = np.array([model_(test[:i]) for i in tqdm(our_range, leave=True, desc="Predicting")])
        print("Done Predicting  !")
        today = test[our_range]
        buys = np.argwhere(predictions > today).flatten()
        sells = np.argwhere(predictwions < today).flatten()
        result = simulator(test, buys, sells, STARTING_FUNDS)
        earnings = result - STARTING_FUNDS
        avg_earnings.append(earnings)
        
        fig, ax = plt.subplots()
        ax.plot(test[int(len(test)*3)+1:], color="blue")
        ax.plot(predictions, color="orange")
        ax.vlines(sells, ax.get_ylim()[0], ax.get_ylim()[1], color="red")
        ax.vlines(buys, ax.get_ylim()[0], ax.get_ylim()[1], color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.show()

        print(f"We earned {earnings}")
        break
    print(f"We made this much money!: {np.mean(avg_earnings)}")


""" def main(df):
    df["Avg"] = (df["High"] + df["Low"])/2
    df["MA"] = df["Avg"].rolling(WINDOW, center=True).mean()
    stdev = df['Avg'].rolling(WINDOW, center=True).std()
    df[f"{STDEV}_stdev"] = stdev*STDEV + df['MA']
    df[f"-{STDEV}_stdev"] = stdev*-STDEV + df['MA']
    df = df.loc[WINDOW//2:]
    #df[]
    # assert len()==len(df), f"{len(df)}, {len(ma)}"
    del df['Volume']
    del df['OpenInt']
    # df['Date'] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce')
    # print(df['Date'])
    # df = df[pd.notnull(df['Date'])]
    # df = df[df['Date'] > 0]
    sells = np.argwhere((df['Avg'] > df[f"{STDEV}_stdev"]).values)
    buys = np.argwhere((df['Avg'] < df[f'-{STDEV}_stdev']).values)

    print(f"Simulation Results: STARTING_FUNDS={STARTING_FUNDS}")
    print(simulator(df['Avg'].values, buys=buys, sells=sells, pocket=STARTING_FUNDS))

    ax = df.plot(title=filename, logy=True)
    ax.vlines(sells, ax.get_ylim()[0], ax.get_ylim()[1], color="red")
    ax.vlines(buys, ax.get_ylim()[0], ax.get_ylim()[1], color="green")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show() """

if __name__ == "__main__":
    main()