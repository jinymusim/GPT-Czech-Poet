import csv
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

FILENAME = os.path.join(os.path.dirname(__file__), 'significance_test_data.csv')

def display_comparision(comparision_header: str, rhyme_data: np.ndarray = None, meter_data: np.ndarray = None, year_data:np.ndarray =None, 
                        rhyme_base:float = None, rhyme_improved:float = None, meter_base:float = None, meter_improved:float = None, 
                        year_base:float = None, year_improved:float = None, p_value_checks:list = [1, 5, 25, 50, 75, 95, 99]):
    # CMAP
    _CMAP = plt.get_cmap('Dark2')
    SLICED_CMAP = _CMAP(np.linspace(0, 1, len(p_value_checks) + 1))

    # Plot Rhyme Data
    if not rhyme_data is  None:
        sns.histplot(data=pd.DataFrame({'rhyme_acc': rhyme_data}), x='rhyme_acc').set(title=f'RHYME {comparision_header}')
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
    if  not meter_data is None:
        sns.histplot(data=pd.DataFrame({'meter_acc': meter_data}), x='meter_acc').set(title=f'METER {comparision_header}')
        meter_lows_pos = np.percentile(meter_data, p_value_checks)
        for i, (p_value, value) in enumerate(zip(p_value_checks, meter_lows_pos)):
            plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
        if meter_base != None and meter_improved != None:
            percentiles = stats.percentileofscore(meter_data, [meter_base, meter_improved])
            plt.axvline(x=meter_base, color='r' , ls='--', label=f'BASE: {percentiles[0]} %')
            plt.axvline(x=meter_improved, color='black' , ls='--', label=f'IMPROVED: {percentiles[1]} %')
        plt.legend()
        plt.show()
        plt.figure()

    # Plot Year Data
    if not year_data is None:
        sns.histplot(data=pd.DataFrame({'year_acc': year_data}), x='year_acc').set(title=f'YEAR {comparision_header}')
        year_lows_pos = np.percentile(year_data, p_value_checks)
        for i, (p_value, value) in enumerate(zip(p_value_checks, year_lows_pos)):
            plt.axvline(x=value, color= SLICED_CMAP[i], ls='--', label= f'{p_value} %')
        if year_base != None and year_improved != None:
            percentiles = stats.percentileofscore(year_data, [year_base, year_improved])
            plt.axvline(x=year_base, color='r' , ls='--', label=f'BASE: {percentiles[0]} %')
            plt.axvline(x=year_improved, color='black' , ls='--', label=f'IMPROVED: {percentiles[1]} %')
        plt.legend()
        plt.show()
        plt.figure()



with open(FILENAME, 'r') as file:
    rows = []
    spamreader = csv.reader(file, delimiter=',')
    for row in spamreader:
        rows.append(row)
    # Display RHYME Distil RAW to Disil Syllable 
    display_comparision(comparision_header= rows[0][1], 
                        rhyme_data=np.asarray( list(map(float,rows[0][2:])) ), rhyme_base=0.9682, rhyme_improved=0.9689)
    # Display RHYME RobeCzech RAW to RobeCzech Syllable 
    display_comparision(comparision_header= rows[1][1],
                        rhyme_data=np.asarray( list(map(float,rows[1][2:])) ), rhyme_base=0.4806, rhyme_improved=0.9468)
    # Display RHYME RobeCzech SYLLABLE to Disil Syllable 
    display_comparision(comparision_header= rows[2][1],
                        rhyme_data=np.asarray( list(map(float,rows[2][2:])) ), rhyme_base=0.9468, rhyme_improved=0.9689)
    # Display YEAR RobeCzech SYLLABLE to RobeCzech Raw 
    display_comparision(comparision_header= rows[3][1],
                        year_data=np.asarray( list(map(float,rows[3][2:])) ), year_base=0.4255, year_improved=0.4745)
    # Display YEAR Roberta Raw to RobeCzech Raw 
    display_comparision(comparision_header= rows[4][1],
                        year_data=np.asarray( list(map(float,rows[4][2:])) ), year_base=0.4315, year_improved=0.4745)
    # Display RHYME Roberta Raw Unweighted to Roberta Raw Weighted
    display_comparision(comparision_header= rows[5][1],
                        rhyme_data=np.asarray( list(map(float,rows[5][2:])) ), rhyme_base=0.9678, rhyme_improved=0.9693)
    # Display METER Distil RAW to Disil Syllable 
    display_comparision(comparision_header= rows[6][1],
                        meter_data=np.asarray( list(map(float,rows[6][2:])) ), meter_base=0.8987, meter_improved=0.8952)
    # Display METER Distil Syllable to Disil Syllable Context
    display_comparision(comparision_header= rows[7][1],
                        meter_data=np.asarray( list(map(float,rows[7][2:])) ), meter_base=0.8952, meter_improved=0.9494)
    # Display METER Roberta Syllable Context to Disil Syllable Context
    display_comparision(comparision_header= rows[8][1],
                        meter_data=np.asarray( list(map(float,rows[8][2:])) ), meter_base=0.9434, meter_improved=0.9494)
    # Display METER Roberta Raw Unweighted to Roberta Raw Weighted
    display_comparision(comparision_header= rows[9][1],
                        meter_data=np.asarray( list(map(float,rows[9][2:])) ), meter_base=0.8973, meter_improved=0.8928)
    # Display RHYME and METER Meter_verse input base and pretarined
    display_comparision(comparision_header= rows[10][1],
                        rhyme_data=np.asarray( list(map(float,rows[10][2:])) ), rhyme_base=0.4706, rhyme_improved=0.4977, 
                        meter_data=np.asarray( list(map(float,rows[11][2:])) ), meter_base=0.9461, meter_improved=0.9483)
    # Display RHYME and METER BASIC input base and pretarined
    display_comparision(comparision_header= rows[12][1],
                        rhyme_data=np.asarray( list(map(float,rows[12][2:])) ), rhyme_base=0.4073, rhyme_improved=0.3355, 
                        meter_data=np.asarray( list(map(float,rows[13][2:])) ), meter_base=0.9400, meter_improved=0.9436)
    # Display RHYME BASE tok Basic to Forced Generation
    display_comparision(comparision_header= rows[14][1],
                        rhyme_data=np.asarray( list(map(float,rows[14][2:])) ), rhyme_base=0.5051, rhyme_improved=0.5206,
                        meter_data=np.asarray( list(map(float,rows[15][2:])) ), meter_base=0.9520, meter_improved=0.9092)
    # Display RHYME OUR tok Basic to Forced Generation
    display_comparision(comparision_header= rows[16][1],
                        rhyme_data=np.asarray( list(map(float,rows[16][2:])) ), rhyme_base=0.4776, rhyme_improved=0.4843,
                        meter_data=np.asarray( list(map(float,rows[17][2:])) ), meter_base=0.9533, meter_improved=0.9114)
    # Display RHYME SYLLABLE tok Basic to Forced Generation
    display_comparision(comparision_header= rows[18][1],
                        rhyme_data=np.asarray( list(map(float,rows[18][2:])) ), rhyme_base=0.6909, rhyme_improved=0.6925)
    # Display RHYME BASIC Input to METER\_VERSE Input
    display_comparision(comparision_header= rows[19][1],
                        rhyme_data=np.asarray( list(map(float,rows[19][2:])) ), rhyme_base=0.4073, rhyme_improved=0.4956)
    # Display RHYME BASE Tok to OUR Tok
    display_comparision(comparision_header= rows[20][1],
                        rhyme_data=np.asarray( list(map(float,rows[20][2:])) ), rhyme_base=0.5206, rhyme_improved=0.4843)
    # Display RHYME UNICODE Tok Basic to Forced Gen 
    display_comparision(comparision_header= rows[21][1],
                        rhyme_data=np.asarray( list(map(float,rows[21][2:])) ), rhyme_base=0.6357, rhyme_improved=0.8128)
    # Display RHYME BASE Tok to SYLLABLE Tok
    display_comparision(comparision_header= rows[22][1],
                        rhyme_data=np.asarray( list(map(float,rows[22][2:])) ), rhyme_base=0.5206, rhyme_improved=0.6925)