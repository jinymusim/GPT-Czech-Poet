import csv
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

FILENAME = os.path.join(os.path.dirname(__file__), 'significance_test_data.csv')

def display_comparision(comparision_header: str, rhyme_data: np.ndarray, meter_data: np.ndarray, year_data:np.ndarray, 
                        rhyme_base:float = None, rhyme_improved:float = None, meter_base:float = None, meter_improved:float = None, 
                        year_base:float = None, year_improved:float = None, p_value_checks:list = [1, 5, 25, 50, 75, 95, 99]):
    # CMAP
    _CMAP = plt.get_cmap('Dark2')
    SLICED_CMAP = _CMAP(np.linspace(0, 1, len(p_value_checks) + 1))

    # Plot Rhyme Data
    sns.histplot(data=pd.DataFrame({'rhyme_acc': rhyme_data}), x='rhyme_acc').set(title=comparision_header)
    rhyme_lows_pos = np.percentile(rhyme_data, p_value_checks)
    for i, (p_value, value) in enumerate(zip(p_value_checks, rhyme_lows_pos)):
        plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
    if rhyme_base != None and rhyme_improved != None:
        percentiles = stats.percentileofscore(rhyme_data, [rhyme_base, rhyme_improved])
        plt.axvline(x=rhyme_base, color='r' , ls='--', label=f'BASE: {percentiles[0]} %')
        plt.axvline(x=rhyme_improved, color='black' , ls='--', label=f'IMPROVED: {percentiles[1]} %')
    plt.legend()
    plt.show()
    plt.figure()

    # Plot Meter Data
    sns.histplot(data=pd.DataFrame({'meter_acc': meter_data}), x='meter_acc').set(title=comparision_header)
    meter_lows_pos = np.percentile(meter_data, p_value_checks)
    for i, (p_value, value) in enumerate(zip(p_value_checks, meter_lows_pos)):
        plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
    if meter_base != None and meter_improved != None:
        percentiles = stats.percentileofscore(meter_data, [meter_base, meter_improved])
        plt.axvline(x=meter_base, color='r' , ls='--', label=f'BASE: {percentiles[0]} %')
        plt.axvline(x=meter_improved, color='black' , ls='--', label=f'IMPROVED: {percentiles[1]} %')
    plt.legend()
    plt.show()


with open(FILENAME, 'r') as file:
    rows = []
    spamreader = csv.reader(file, delimiter=',')
    for row in spamreader:
        rows.append(row)
    # Display Raw to Pretrained
    display_comparision(comparision_header= rows[0][1], 
                        rhyme_data=np.asarray( list(map(float,rows[0][2:])) ), rhyme_base=0.3544, rhyme_improved=0.5732,
                        meter_data=np.asarray( list(map(float,rows[1][2:])) ), meter_base= 0.8453, meter_improved=0.8537,
                        year_data=np.asarray( list(map(float,rows[2][2:])) ),)