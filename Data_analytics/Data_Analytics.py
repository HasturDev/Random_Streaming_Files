from typing import List
from functools import partial

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

WINDOW = 50
STDEV = 2.5
STARTING_FUNDS = 1000
FILENAMES = list(Path(r"C:\\Users\\i_miz\\Documents\\Visual_Studio_Projects\\Streaming_uploads\\Random_Streaming_Files\\Data_analytics\\Stock_Market_data\\Data\\Stocks").glob('*.txt'))

"""
#REF: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands#:~:text=Bollinger%20Bands%20are%20envelopes%20plotted,Period%20and%20Standard%20Deviations%2C%20StdDev.
Short term: 10 day moving average, bands at 1.5 standard deviations. (1.5 times the standard dev. +/- the SMA)
Medium term: 20 day moving average, bands at 2 standard deviations.
Long term: 50 day moving average, bands at 2.5 standard deviations.
"""

def train_test_split(df, split_date):
    return df[df["Date"] <= split_date], df[df["Date"] > split_date]

def simulator(price: List[float], buys: List[int], sells: List[int], pocket: float) -> float:
    number_shares = 0
    for i, buysell in sorted([(i, 'buy') for i in buys] + [(i, 'sell') for i in sells]):
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
    # print("Starting Model")
    try:
        model = ARIMA(series, order=(int(ar),int(diff),int(mov_avg)))
        #fit model
        model_fit = model.fit(disp=0)
        out = model_fit.forecast()[0]
    except:
        return -1000
    # print("We did it")
    # print(out)
    return out

def get_files():
    assert FILENAMES, "No filenames found"
    for f in FILENAMES[:10]:
        try:
            df = pd.read_csv(f)
        except:
            continue
        df["Avg"] = (df["High"] + df["Low"])/2
        yield df

def objective_parameter_tuning(parameters):
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
    print("Starting HyperParameter Tuning")
    uniform_int = lambda x, y, z: scope.int(hp.quniform(x, y, z, q=1))
    space = {
        'ar': uniform_int('ar', 0, 10),
        'diff': uniform_int('diff', 0, 2),
        'mov_avg': uniform_int('mov_avg', 0, 10)
    }
    best = fmin(objective_parameter_tuning, space, algo=tpe.suggest, max_evals=100)
    print("Done!")
    return best
 
def main():
    parameters = hyperparameter_tuning()
    model_ = partial(model, **parameters)
    avg_earnings = []
    for df in tqdm(get_files(), total=len(FILENAMES)):
        test = df['Avg'].values[int(len(df)*.66):]
        our_range = list(range(int(len(test)*.3), len(test)))
        print("Predicting...")
        predictions = np.array([model_(test[:i]) for i in our_range])
        print("Done Predicting!")
        today = test[our_range]
        buys = np.argwhere(predictions > today)
        sells = np.argwhere(predictions < today)
        result = simulator(test, buys, sells, STARTING_FUNDS)
        earnings = result - STARTING_FUNDS
        avg_earnings.append(earnings)
        print(f"We earned {earnings}")
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