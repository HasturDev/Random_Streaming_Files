from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import os

from subprocess import check_output
from pathlib import Path

WINDOW = 50
STDEV = 2.5
STARTING_FUNDS = 1000

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


def main(filename):
    print(filename)
    df = pd.read_csv(filename)
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
    plt.show()

if __name__ == "__main__":
    #print(check_output(["ls", "C:\Users\i_miz\Documents\Visual_Studio_Projects\Streaming_uploads\Random_Streaming_Files\Data_analytics\Stock_Market_data"]).decode("utf8"))
    #os.chdir('C:\Users\i_miz\Documents\Visual_Studio_Projects\Streaming_uploads\')
    filenames = list(Path(r"C:\\Users\\i_miz\\Documents\\Visual_Studio_Projects\\Streaming_uploads\\Random_Streaming_Files\\Data_analytics\\Stock_Market_data\\Data\\Stocks").glob('*.txt'))
    assert filenames, "No filenames found"
    # filenames = random.sample(filenames,10)
    for f in filenames:
        main(f)
        break